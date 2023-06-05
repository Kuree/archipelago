import os
import sys
import pycyclone
import glob
import csv
import argparse
from PIL import Image, ImageDraw
from .io import load_routing_result

# Draw parameters
GLOBAL_TILE_WIDTH = 100
GLOBAL_TILE_MARGIN = 20
GLOBAL_TILE_WIDTH_INNER = GLOBAL_TILE_WIDTH - 2*GLOBAL_TILE_MARGIN
GLOBAL_OFFSET_X = 10
GLOBAL_OFFSET_Y = 10
GLOBAL_NUM_TRACK = 5
GLOBAL_ARROW_DISTANCE = GLOBAL_TILE_WIDTH_INNER // (GLOBAL_NUM_TRACK*2+1)

# Mapping of numbers
side_map = ["Right", "Bottom", "Left", "Top"]
io_map = ["IN", "OUT"]

def draw_usage(draw, x, y, used_track, total_track=40):
    if used_track == 0:
        return
    elif used_track < 5:
        fill = "Black"
    else:
        fill = "Red"

    x=GLOBAL_OFFSET_X + x*GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    y=GLOBAL_OFFSET_Y + y*GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    w = GLOBAL_TILE_WIDTH - 2*GLOBAL_TILE_MARGIN
    txy = (x + int(w*0.25), y + int(w*0.6))
    text = f"{used_track}/{total_track}"
    draw.text(xy=txy, text=text, fill="Red")

def create_tile(draw, x, y, w=GLOBAL_TILE_WIDTH, tile_type="PE", width=2):
    if tile_type == "PE":
        color_tile = "PapayaWhip"
        color_line = "Black"
        pr = 0.4
    elif tile_type == "MEM":
        color_tile = "Khaki"
        color_line = "Black"
        pr = 0.35
    elif tile_type == "IO":
        color_tile = "White"
        color_line = "Black"
        pr = 0.4
    else:
        print("[Error] unsupported tile type")
        exit()
    px=GLOBAL_OFFSET_X + x*GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    py=GLOBAL_OFFSET_Y + y*GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    pw = GLOBAL_TILE_WIDTH - 2*GLOBAL_TILE_MARGIN
    xy = [(px,py), (px+pw,py), (px+pw,py+pw), (px,py+pw)]
    txy = (px + int(pw*pr), py + int(pw*0.4))
    draw.polygon(xy=xy, fill=color_tile, outline=color_line, width=width)
    draw.text(xy=txy, text=tile_type, fill="Black")
    # draw its coordinate
    cxy = (px + int(pw*0.25), py + int(pw*0.15))
    draw.text(xy=cxy, text=f"({x},{y})", fill="Black")


def draw_arraw(draw, x, y, dir="UP", len=GLOBAL_TILE_MARGIN, color="Black", width=1):
    if dir == "UP":
        dx = 0
        dy = -len
        rx = 0.09
        ry = 0.8
    elif dir == "DOWN":
        dx = 0
        dy = len
        rx = 0.09
        ry = 0.8
    elif dir == "LEFT":
        dx = -len
        dy = 0
        rx = 0.8
        ry = 0.09
    elif dir == "RIGHT":
        dx = len
        dy = 0
        rx = 0.8
        ry = 0.09
    else:
        print("[Error] unsupported arraw direction")
        exit()
    xy = [(x,y), (x+dx, y+dy)]
    if dir == "UP" or dir == "DOWN":
        lxy = [(x+dx, y+dy), (x+int(dy*rx), y+int(dy*ry))]
        rxy = [(x+dx, y+dy), (x+int(-dy*rx), y+int(dy*ry))]
    else:
        lxy = [(x+dx, y+dy), (x+int(dx*rx), y+int(-dx*ry))]
        rxy = [(x+dx, y+dy), (x+int(dx*rx), y+int(dx*ry))]
    draw.line(xy=xy, fill=color, width=width)
    draw.line(xy=lxy, fill=color, width=width)
    draw.line(xy=rxy, fill=color, width=width)

def draw_arraw_on_tile(draw, tile_x, tile_y, side="Top", io="IN", track_id=0, color="Black", width=1):
    if side=="Top":
        if io=="IN":
            dir = "DOWN"
            x = GLOBAL_OFFSET_X + GLOBAL_TILE_MARGIN + tile_x*GLOBAL_TILE_WIDTH + (track_id+1)*GLOBAL_ARROW_DISTANCE
            y = GLOBAL_OFFSET_Y + tile_y*GLOBAL_TILE_WIDTH
        elif io=="OUT":
            dir = "UP"
            x = GLOBAL_OFFSET_X + GLOBAL_TILE_MARGIN + tile_x*GLOBAL_TILE_WIDTH + (track_id+1+GLOBAL_NUM_TRACK)*GLOBAL_ARROW_DISTANCE
            y = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y*GLOBAL_TILE_WIDTH
    elif side=="Right":
        if io=="IN":
            dir = "LEFT"
            x = GLOBAL_OFFSET_X + tile_x*GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
            y = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y*GLOBAL_TILE_WIDTH + (track_id+1)*GLOBAL_ARROW_DISTANCE
        elif io=="OUT":
            dir = "RIGHT"
            x = GLOBAL_OFFSET_X + tile_x*GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH - GLOBAL_TILE_MARGIN
            y = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y*GLOBAL_TILE_WIDTH + (track_id+1+GLOBAL_NUM_TRACK)*GLOBAL_ARROW_DISTANCE
    elif side=="Bottom":
        if io=="IN":
            dir = "UP"
            x = GLOBAL_OFFSET_X + GLOBAL_TILE_MARGIN + tile_x*GLOBAL_TILE_WIDTH + (track_id+1+GLOBAL_NUM_TRACK)*GLOBAL_ARROW_DISTANCE
            y = GLOBAL_OFFSET_Y + tile_y*GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
        elif io=="OUT":
            dir = "DOWN"
            x = GLOBAL_OFFSET_X + GLOBAL_TILE_MARGIN + tile_x*GLOBAL_TILE_WIDTH + (track_id+1)*GLOBAL_ARROW_DISTANCE
            y = GLOBAL_OFFSET_Y + tile_y*GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH - GLOBAL_TILE_MARGIN
    elif side=="Left":
        if io=="IN":
            dir = "RIGHT"
            x = GLOBAL_OFFSET_X + tile_x*GLOBAL_TILE_WIDTH
            y = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y*GLOBAL_TILE_WIDTH + (track_id+1+GLOBAL_NUM_TRACK)*GLOBAL_ARROW_DISTANCE
        elif io=="OUT":
            dir = "LEFT"
            x = GLOBAL_OFFSET_X + tile_x*GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
            y = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y*GLOBAL_TILE_WIDTH + (track_id+1)*GLOBAL_ARROW_DISTANCE
    draw_arraw(draw=draw, x=x, y=y, dir=dir, color=color, width=width)

def get_array_size_from_graph(graph):
    """
    There should be a better way of doing this...
    """
    array_width = 0
    array_height = 0
    for x, y in graph:
        if (x+1) > array_width:
            array_width = x+1
        if (y+1) > array_height:
            array_height = y+1 
    return array_width, array_height

def load_graph(graph_files):
    graph_result = {}
    for graph_file in graph_files:
        bit_width = os.path.splitext(graph_file)[0]
        bit_width = int(os.path.basename(bit_width))
        graph = pycyclone.io.load_routing_graph(graph_file)
        graph_result[bit_width] = graph
    return graph_result

def draw_all_tiles(draw, graph):
    for x, y in graph:
        switchbox = graph[x, y].switchbox
        sides = switchbox.SIDES
        # draw the tiles
        if y == 0:
            create_tile(draw=draw, x=x, y=y, tile_type="IO")
        elif (x+1)%4==0:
            create_tile(draw=draw, x=x, y=y, tile_type="MEM")
        else:
            create_tile(draw=draw, x=x, y=y, tile_type="PE")
        # draw the arraws
        for i in range(sides):
            side = pycyclone.SwitchBoxSide(i)
            sbs = switchbox.get_sbs_by_side(side)
            for io in ["IN", "OUT"]:
                for i in range(len(sbs)):
                    draw_arraw_on_tile(draw, tile_x=x, tile_y=y, side=side.name, io=io, track_id=i%GLOBAL_NUM_TRACK)

def draw_used_tracks(draw, routing_result, target_bitwidth):
    for netlist in routing_result.values():
        for net in netlist:
            for node in net:
                if node[0] == "SB":
                    # need to parse the SB logic
                    track, x, y, side, io_, bit_width = node[1:]
                    if bit_width == target_bitwidth:
                        draw_arraw_on_tile(draw, tile_x=x, tile_y=y, side=side_map[side], io=io_map[io_], track_id=track, color="Red", width=3)


def collect_track_usage(routing_result, array_width, array_height):
    result = {}
    # initialize the result
    for bit_width in [1, 16]:
        result[bit_width] = {}
        for x in range(array_width):
            for y in range(array_height):
                result[bit_width][(x,y)] = {}
                for side in side_map:
                    result[bit_width][(x,y)][side] = {}
                    for io in io_map:
                        result[bit_width][(x,y)][side][io] = {}
                        for track in range(GLOBAL_NUM_TRACK):
                            result[bit_width][(x,y)][side][io][track] = 0

    for netlist in routing_result.values():
        for net in netlist:
            for node in net:
                if node[0] == "SB":
                    # need to parse the SB logic
                    track, x, y, side, io_, bit_width = node[1:]
                    # if result[bit_width][(x,y)][side_map[side]][io_map[io_]][track] == 1:
                    #     print(f"[Warning] duplicated track ({bit_width}bit):")
                    #     print(f"(x,y) = ({x},{y})")
                    #     print(f"(side, IO) = ({side_map[side]}, {io_map[io_]}")
                    #     print(f"track ID = {track}")
                    #     print("==========================")
                    result[bit_width][(x,y)][side_map[side]][io_map[io_]][track] = 1
    return result

def report_track_usage(bit_width, width, height, track_usage):
    total_horizontal_tracks = (GLOBAL_NUM_TRACK*4)*(width-1)*(height-1)
    total_vertical_tracks = (GLOBAL_NUM_TRACK*4)*(width)*(height-2)
    used_horizontal_tracks = 0
    used_vertical_tracks = 0
    for x in range(width):
        for y in range(height):
            if y == 0: # skip the IO tile
                continue
            for side in side_map:
                if x==0 and side=="Left":
                    continue
                elif x==(width-1) and side=="Right":
                    continue
                elif y==1 and side=="Top":
                    continue
                elif y==(height-1) and side=="Bottom":
                    continue
                for io in io_map:
                    for track in range(GLOBAL_NUM_TRACK):
                        used = track_usage[bit_width][(x,y)][side][io][track]
                        if side in ["Right", "Left"]:
                            used_horizontal_tracks += used
                        elif side in ["Top", "Bottom"]:
                            used_vertical_tracks += used
    total_tracks = total_horizontal_tracks + total_vertical_tracks
    used_tracks = used_horizontal_tracks + used_vertical_tracks
    print(f"============== Track Usage : {bit_width}-bit ==============")
    print(f"Track Usage (H)   = {used_horizontal_tracks}/{total_horizontal_tracks} ({100*used_horizontal_tracks/total_horizontal_tracks:.2f}%)")
    print(f"Track Usage (V)   = {used_vertical_tracks}/{total_vertical_tracks} ({100*used_vertical_tracks/total_vertical_tracks:.2f}%)")
    print(f"Total Track Usage = {used_tracks}/{total_tracks} ({100*used_tracks/total_tracks:.2f}%)")
    print()


def main():
    parser = argparse.ArgumentParser("Congestion Map Dumper")
    parser.add_argument("-i,-d", action="store", type=str, required=True, dest="input_dir",
                        help="PnR collateral directory, typically it should be /aha/garnet/temp/")
    parser.add_argument("--dump-congestion-map", action="store_true", dest="dump_congestion_map",
                        help="whether to dump congestion map or not")
    args = parser.parse_args()
    data_dir = args.input_dir
    dump_congestion_map = args.dump_congestion_map
    assert os.path.isdir(data_dir), data_dir + " is not a directory"
    files = glob.glob(os.path.join(data_dir, "*"))
    assert len(files) > 0, data_dir + " is empty"
    # find out useful files
    graph_files = [x for x in files if os.path.splitext(x)[-1] == ".graph"]
    assert len(graph_files) > 0, "Unable to find any routing graph from " + data_dir
    route_file = [x for x in files if os.path.splitext(x)[-1] == ".route"]
    assert len(route_file) == 1, "Only one PnR route result can be in the data folder"
    route_file = route_file[0]

    # load the graph
    graphs = load_graph(graph_files)

    # find the array size
    array_width, array_height = get_array_size_from_graph(graphs[1])

    # load raw routing result
    raw_routing_result = load_routing_result(route_file)

    # do analysis
    track_usage = collect_track_usage(raw_routing_result, array_width, array_height)

    # do the 1bit and 16bit seperately
    for width, graph in graphs.items():
        if dump_congestion_map:
            # initialize image
            img_width = array_width*GLOBAL_TILE_WIDTH + 3*GLOBAL_OFFSET_X
            img_height = array_height*GLOBAL_TILE_WIDTH + 3*GLOBAL_OFFSET_X
            img = Image.new('RGB', (img_width, img_height), "White")
            draw = ImageDraw.Draw(img)
            # draw all the tiles
            draw_all_tiles(draw, graph)
            # color the used tracks
            draw_used_tracks(draw, raw_routing_result, width)
            # draw usage
            for x in range(array_width):
                for y in range(array_height): 
                    if y == 0:# skip the IO tile
                        continue
                    tile_track_usage = 0
                    for side in side_map:
                        for io in io_map:
                            for track in range(GLOBAL_NUM_TRACK):
                                tile_track_usage += track_usage[width][(x,y)][side][io][track]
                    draw_usage(draw=draw, x=x, y=y, used_track=tile_track_usage, total_track=40)
            # save the congestion map
            img.save(f"bin/congestion_{width}bit.png", format="PNG")
        # report tatal track usage
        print()
        report_track_usage(bit_width=width, width=array_width, height=array_height, track_usage=track_usage)



if __name__ == "__main__":
    main()
