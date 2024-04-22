import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox
from deepsvg.svgtensor_dataset import SVGTensorDataset, load_dataset
from configs.deepsvg.hierarchical_ordered import Config
from deepsvg.difflib.tensor import SVGTensor
from deepsvg import utils

# utils.set_seed(2022)

cfg = Config()
train, valid = load_dataset(cfg)


def draw_svg(idx):
    data = train.get(idx, [*cfg.model_args, 'tensor_grouped'])
    svg_tensor = data['tensor_grouped'][0].copy().drop_sos().unpad()
    svg = SVG.from_tensor(svg_tensor.data, viewbox=Bbox(256)).normalize()
    svg.draw(do_display=False, file_path='test.svg')


draw_svg(10)

def draw_png(idx):
    data = train.get(idx, [*cfg.model_args])
    print(data['args'].shape, data['commands'].shape)
    commands = data['commands']
    args = data['args']

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)

    for path_commands, path_args in zip(commands, args):
        verts = []
        codes = []
        for command, args in zip(path_commands, path_args):
            cmd = SVGTensor.COMMANDS_SIMPLIFIED[int(command)]
            
            # MOVETO
            if cmd == 'm':
                x = args[-2].item()
                y = args[-1].item()
                verts.append((x, 256 - y))
                codes.append(matplotlib.path.Path.MOVETO)
            
            # LINETO
            elif cmd == 'l':
                x = args[-2].item()
                y = args[-1].item()
                verts.append((x, 256 - y))
                codes.append(matplotlib.path.Path.LINETO)

            # CUBIC BEZIER
            elif cmd == 'c':
                qx1 = args[-6].item()
                qy1 = args[-5].item()
                qx2 = args[-4].item()
                qy2 = args[-3].item()
                x = args[-2].item()
                y = args[-1].item()                    
                verts.append((qx1, 256 - qy1))
                codes.append(matplotlib.path.Path.CURVE4)
                verts.append((qx2, 256 - qy2))
                codes.append(matplotlib.path.Path.CURVE4)
                verts.append((x, 256 - y))
                codes.append(matplotlib.path.Path.CURVE4)
            
            # CLOSE POLY
            elif cmd == 'z':
                # z doesn't appear to be in the data
                verts.append((0, 0))
                codes.append(matplotlib.path.Path.CLOSEPOLY)
        
        if len(verts) > 0:
            path = matplotlib.path.Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2)
            ax.add_patch(patch)

    # plt.show()
    plt.savefig('test.png')