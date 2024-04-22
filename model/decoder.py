from .layers.transformer import *
from .layers.improved_transformer import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

"""
0: SVG END
1: MASK
2: EOM
3: FACE_END
4: LOOP END
5: CMD  END
"""
NUM_MASK_AND_EOM = 2
MASK = 1
EOM = 2
NUM_END_TOKEN = 3
CAUSAL_PAD = 3
PIX_PAD = NUM_END_TOKEN + NUM_MASK_AND_EOM
COORD_PAD = NUM_END_TOKEN + NUM_MASK_AND_EOM
SVG_END = 1
CMD_END = 2 + NUM_MASK_AND_EOM
LOOP_END = 1 + NUM_MASK_AND_EOM
FACE_END = 0 + NUM_MASK_AND_EOM
CMD_END_P = [CMD_END, CMD_END]
LOOP_END_P = [LOOP_END, LOOP_END]
FACE_END_P = [FACE_END, FACE_END]

# FIGR-SVG-svgo
BBOX = 200

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class SketchDecoder(nn.Module):
  """
  Autoregressive generative model 
  """

  def __init__(self,
               config,
               pix_len,
               text_len,
               num_text_token,
               word_emb_path=None,
               pos_emb_path=None):
    """
    Initializes FaceModel.
    """
    super(SketchDecoder, self).__init__()
    self.pix_len = pix_len
    self.embed_dim = config['embed_dim']

    self.text_len = text_len
    self.num_text_token = num_text_token
    self.num_image_token = BBOX * BBOX + PIX_PAD + SVG_END
    self.total_token = num_text_token + self.num_image_token
    self.total_seq_len = text_len + pix_len
    self.loss_img_weight = 7

    seq_range = torch.arange(self.total_seq_len)
    logits_range = torch.arange(self.total_token)

    seq_range = rearrange(seq_range, 'n -> () n ()')
    logits_range = rearrange(logits_range, 'd -> () () d')

    logits_mask = (
        ((seq_range >= text_len) & (logits_range < num_text_token)) |
        ((seq_range < text_len) & (logits_range >= num_text_token))
    )

    self.register_buffer('logits_mask', logits_mask, persistent=False)
    # Sketch encoders
    self.coord_embed_x = Embedder(BBOX+COORD_PAD+SVG_END, self.embed_dim, padding_idx=MASK)
    self.coord_embed_y = Embedder(BBOX+COORD_PAD+SVG_END, self.embed_dim, padding_idx=MASK)

    self.pixel_embed = Embedder(self.num_image_token, self.embed_dim, padding_idx=MASK)
    self.pos_embed = PositionalEncoding(max_len=self.total_seq_len, d_model=self.embed_dim)
    self.logit_fc = nn.Linear(self.embed_dim, self.total_token)
    
    decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, 
                        dim_feedforward= config['hidden_dim'], nhead=config['num_heads'], dropout=config['dropout_rate'])
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, config['num_layers'], decoder_norm)

    assert word_emb_path is not None, 'text_emb_dir must be provided'
    if word_emb_path is not None:
      self.text_emb = nn.Embedding.from_pretrained(torch.load(word_emb_path, map_location='cpu'))
      assert self.embed_dim == self.text_emb.weight.shape[1], 'emb_dim must match pretrained text embedded dim'
    if pos_emb_path is not None:
      self.text_pos_emb = nn.Embedding.from_pretrained(torch.load(pos_emb_path, map_location='cpu'))
      assert self.embed_dim == self.text_pos_emb.weight.shape[1], 'emb_dim must match pretrained text embedded dim'

  def forward(self, pix, xy, mask, text, return_loss=False):
    '''
    pix.shape  [batch_size, max_len]
    xy.shape   [batch_size, max_len, 2]
    mask.shape [batch_size, max_len]
    text.shape [batch_size, text_len]
    '''
    pixel_v = pix[:, :-1] if return_loss else pix
    xy_v = xy[:, :-1] if return_loss else xy
    pixel_mask = mask[:, :-1] if return_loss else mask

    c_bs, c_seqlen, device = text.shape[0], text.shape[1], text.device
    if pixel_v[0] is not None:
      c_seqlen += pixel_v.shape[1]  

    # Context embedding values
    context_embedding = torch.zeros((1, c_bs, self.embed_dim)).to(device) # [1, bs, dim]

    # tokens.shape [batch_size, text_len, emb_dim]
    tokens = self.text_emb(text)

    # Data input embedding
    if pixel_v[0] is not None:
      # coord_embed.shape [batch_size, max_len-1, emb_dim]
      # pixel_embed.shape [batch_size, max_len-1, emb_dim] 
      coord_embed = self.coord_embed_x(xy_v[...,0]) + self.coord_embed_y(xy_v[...,1]) # [bs, vlen, dim]
      pixel_embed = self.pixel_embed(pixel_v)
      embed_inputs = pixel_embed + coord_embed

      # tokens.shape [batch_size, text_len+max_len-1, emb_dim]
      tokens = torch.cat((tokens, embed_inputs), dim=1)

    # embeddings.shape [text_len+1 or text_len+max_len, batch_size, emb_dim]
    embeddings = torch.cat([context_embedding, tokens.transpose(0,1)], axis=0)
    decoder_inputs = self.pos_embed(embeddings) 

    memory_encode = torch.zeros((1, c_bs, self.embed_dim)).to(device)
    
    # nopeak_mask.shape [c_seqlen+1, c_seqlen+1]
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(c_seqlen+1).to(device)  # masked with -inf
    if pixel_mask is not None:
      # pixel_mask.shape [batch_size, text_len+max_len]
      pixel_mask = torch.cat([(torch.zeros([c_bs, context_embedding.shape[0]+self.text_len])==1).to(device), pixel_mask], axis=1)  
    decoder_out = self.decoder(tgt=decoder_inputs, memory=memory_encode, memory_key_padding_mask=None,
                               tgt_mask=nopeak_mask, tgt_key_padding_mask=pixel_mask)

    # Logits fc
    logits = self.logit_fc(decoder_out)  # [seqlen, bs, dim] 
    logits = logits.transpose(1,0)  # [bs, textlen+seqlen, total_token] 

    logits_mask = self.logits_mask[:, :c_seqlen+1]
    max_neg_value = -torch.finfo(logits.dtype).max
    logits.masked_fill_(logits_mask, max_neg_value)

    if return_loss:
      logits = rearrange(logits, 'b n c -> b c n')
      text_logits = logits[:, :, :self.text_len]
      pix_logits = logits[:, :, self.text_len:]

      pix_logits = rearrange(pix_logits, 'b c n -> (b n) c')
      pix_mask = ~mask.reshape(-1)
      pix_target = pix.reshape(-1) + self.num_text_token

      text_loss = F.cross_entropy(text_logits, text)
      pix_loss = F.cross_entropy(pix_logits[pix_mask], pix_target[pix_mask], ignore_index=MASK+self.num_text_token)
      loss = (text_loss + self.loss_img_weight * pix_loss) / (self.loss_img_weight + 1)
      return loss, pix_loss, text_loss
    else:
      return logits
    

  def sample(self, n_samples, text, pixel_seq=None, xy_seq=None):
    """ sample from distribution (top-k, top-p) """
    pix_samples = []
    xy_samples = []
    # latent_ext_samples = []
    top_k = 0
    top_p = 0.5

    # Mapping from pixel index to xy coordiante
    pixel2xy = {}
    x=np.linspace(0, BBOX-1, BBOX)
    y=np.linspace(0, BBOX-1, BBOX)
    xx,yy=np.meshgrid(x,y)
    xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
    for pixel, xy in enumerate(xy_grid):
      pixel2xy[pixel] = xy+COORD_PAD+SVG_END

    # Sample per token
    text = text[:, :self.text_len]
    pixlen = 0 if pixel_seq is None else pixel_seq.shape[1]
    for k in range(text.shape[1] + pixlen, self.total_seq_len):
      if k == text.shape[1]:
        pixel_seq = [None] * n_samples
        xy_seq = [None, None] * n_samples
      
      # pass through model
      with torch.no_grad():
        p_pred = self.forward(pixel_seq, xy_seq, None, text)
        p_logits = p_pred[:, -1, :]

      next_pixels = []
      # Top-p sampling of next pixel
      for logit in p_logits: 
        filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
        next_pixel = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
        next_pixel -= self.num_text_token
        next_pixels.append(next_pixel.item())

      # Convert pixel index to xy coordinate
      next_xys = []
      for pixel in next_pixels:
        if pixel >= PIX_PAD+SVG_END:
          xy = pixel2xy[pixel-PIX_PAD-SVG_END]
        else:
          xy = np.array([pixel, pixel]).astype(int)
        next_xys.append(xy)
      next_xys = np.vstack(next_xys)  # [BS, 2]
      next_pixels = np.vstack(next_pixels)  # [BS, 1]
        
      # Add next tokens
      nextp_seq = torch.LongTensor(next_pixels).view(len(next_pixels), 1).cuda()
      nextxy_seq = torch.LongTensor(next_xys).unsqueeze(1).cuda()
      
      if pixel_seq[0] is None:
        pixel_seq = nextp_seq
        xy_seq = nextxy_seq
      else:
        pixel_seq = torch.cat([pixel_seq, nextp_seq], 1)
        xy_seq = torch.cat([xy_seq, nextxy_seq], 1)
      
      # Early stopping
      done_idx = np.where(next_pixels==0)[0]
      if len(done_idx) > 0:
        done_pixs = pixel_seq[done_idx] 
        done_xys = xy_seq[done_idx]
        # done_ext = latent_ext[done_idx]
       
        # for pix, xy, ext in zip(done_pixs, done_xys, done_ext):
        for pix, xy in zip(done_pixs, done_xys):
          pix = pix.detach().cpu().numpy()
          xy = xy.detach().cpu().numpy()
          pix_samples.append(pix)
          xy_samples.append(xy)
          # latent_ext_samples.append(ext.unsqueeze(0))
  
      left_idx = np.where(next_pixels!=0)[0]
      if len(left_idx) == 0:
        break # no more jobs to do
      else:
        pixel_seq = pixel_seq[left_idx]
        xy_seq = xy_seq[left_idx]
        text = text[left_idx]
    
    # return pix_samples, latent_ext_samples
    return xy_samples

  def fill(self, n_samples, text):
      """ sample from distribution (top-k, top-p) """
      pix_samples = []
      xy_samples = []
      top_k = 0
      top_p = 0.5

      # Mapping from pixel index to xy coordiante
      pixel2xy = {}
      x=np.linspace(0, BBOX-1, BBOX)
      y=np.linspace(0, BBOX-1, BBOX)
      xx,yy=np.meshgrid(x,y)
      xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
      for pixel, xy in enumerate(xy_grid):
        pixel2xy[pixel] = xy+COORD_PAD+SVG_END

      text = text[:, :self.text_len]
      # Sample per token
      pixel_seq = torch.tensor([[
        1
      ]]).cuda()
      xy_seq = torch.tensor([[
        [1,1]
      ]]).cuda()

      start_pos = 0 if pixel_seq is None else pixel_seq.shape[1]
      for k in range(start_pos, self.pix_len):
        if k == 0:
          pixel_seq = [None] * n_samples
          xy_seq = [None, None] * n_samples
        
        # pass through model
        with torch.no_grad():
          p_pred = self.forward(pixel_seq, xy_seq, None, text)
          p_logits = p_pred[:, -1, :]

        next_pixels = []
        # Top-p sampling of next pixel
        for logit in p_logits: 
          filtered_logits = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
          next_pixel = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
          next_pixel -= self.num_text_token
          next_pixels.append(next_pixel.item())

        # Convert pixel index to xy coordinate
        next_xys = []
        for pixel in next_pixels:
          if pixel >= PIX_PAD+SVG_END:
            xy = pixel2xy[pixel-PIX_PAD-SVG_END]
          else:
            xy = np.array([pixel, pixel]).astype(int)
          next_xys.append(xy)
        next_xys = np.vstack(next_xys)  # [BS, 2]
        next_pixels = np.vstack(next_pixels)  # [BS, 1]
          
        # Add next tokens
        nextp_seq = torch.LongTensor(next_pixels).view(len(next_pixels), 1).cuda()
        nextxy_seq = torch.LongTensor(next_xys).unsqueeze(1).cuda()
        
        if pixel_seq[0] is None:
          pixel_seq = nextp_seq
          xy_seq = nextxy_seq
        else:
          pixel_seq = torch.cat([pixel_seq, nextp_seq], 1)
          xy_seq = torch.cat([xy_seq, nextxy_seq], 1)
        
        # Early stopping
        done_idx = np.where(next_pixels==EOM)[0]
        if len(done_idx) > 0:
          done_pixs = pixel_seq[done_idx] 
          done_xys = xy_seq[done_idx]
        
          for pix, xy in zip(done_pixs, done_xys):
            pix = pix.detach().cpu().numpy()
            mask_index1, mask_index2 = (pix==MASK).nonzero()[0]
            left_span = pix[:mask_index1]
            right_span = pix[mask_index1+1 : mask_index2]
            mid_span = pix[mask_index2+1:-1]  # ignore EOM
            pix = np.concatenate([left_span, mid_span, right_span])

            # xy = xy.detach().cpu().numpy()
            xys = []
            for p in pix:
              if p >= PIX_PAD+SVG_END:
                xy = pixel2xy[p-PIX_PAD-SVG_END]
              else:
                xy = np.array([p, p]).astype(int)
              xys.append(xy)
            xys = np.array(xys)
            pix_samples.append(pix)
            xy_samples.append(xys)
    
        left_idx = np.where(next_pixels!=EOM)[0]
        if len(left_idx) == 0:
          break # no more jobs to do
        else:
          pixel_seq = pixel_seq[left_idx]
          xy_seq = xy_seq[left_idx]
          text = text[left_idx]
      
      return xy_samples