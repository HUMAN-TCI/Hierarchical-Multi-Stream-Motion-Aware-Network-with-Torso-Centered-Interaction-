# import argparse
# import os
# import sys

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('video_dir', type=str, help='Directory where videos are stored.')
#     parser.add_argument('--layout', default='linear', help='Which layout to use to position videos in the canvas')
#     parser.add_argument('--how_many', type=int, default=10, help='How many videos to use')
#     args = parser.parse_args()

#     # Check if directory exists
#     if not os.path.exists(args.video_dir):
#         print(f"[ERROR] Directory does not exist: {args.video_dir}")
#         sys.exit(1)

#     print(f"[INFO] Reading from directory: {args.video_dir}")
#     all_files = os.listdir(args.video_dir)

#     # Safely filter for MP4 files
#     video_paths = [os.path.join(args.video_dir, f) for f in all_files if f.endswith('.npy..gif') or f.endswith('.gif')]
    
#     if not video_paths:
#         print(f"[ERROR] No video files (.gif) found in {args.video_dir}")
#         sys.exit(1)

#     try:
#         video_paths = sorted(video_paths, key=lambda x: int(os.path.basename(x).split('_')[0]))
#     except Exception as e:
#         print(f"[WARNING] Could not sort files based on prefix: {e}")
#         video_paths = sorted(video_paths)

#     how_many_vids = min(len(video_paths), args.how_many)
#     video_paths = video_paths[:how_many_vids]

#     # Generate layout
#     if args.layout == 'linear':
#         layout = ["0_0"] + ['+'.join(['w'+str(i) for i in range(j)])+'_0' for j in range(1, how_many_vids)]
#     elif args.layout == 'grid':
#         layout = ["0_0", "w0_0", "w0+w1_0", "w0+w1+w2_0", 
#                   "0_h0", "w4_h0", "w4+w5_h0", "w4+w5+w6_h0", 
#                   "0_h0+h4", "w8_h0+h4", "w8+w9_h0+h4", "w8+w9+w10_h0+h4",
#                   "0_h0+h4+h8", "w12_h0+h4+h8", "w12+w13_h0+h4+h8", "w12+w13+w14_h0+h4+h8"]

#     # Read description
#     desc_path = os.path.join(args.video_dir, 'desc.txt')
#     if not os.path.exists(desc_path):
#         print(f"[WARNING] Description file not found: {desc_path}")
#         desc, rank = "No description available", "N/A"
#     else:
#         with open(desc_path, 'r') as f:
#             lines = f.read().splitlines()
#             if len(lines) >= 2:
#                 desc, rank = lines[0], lines[1]
#             else:
#                 desc, rank = lines[0], "N/A"

#     # Construct ffmpeg command
#     inputs = " ".join([f"-i \"{v}\"" for v in video_paths])
#     input_stream_names = "".join([f"[{i}:v]" for i in range(how_many_vids)])
#     layout_expr = "|".join(layout[:how_many_vids])
#     output_file = os.path.join(args.video_dir, 'output.gif')
#     text = f"{desc} (Rank of correct result = {rank})"

#     cmd = (
#         f"ffmpeg {inputs} -y -filter_complex "
#         f"\"{input_stream_names}xstack=inputs={how_many_vids}:layout={layout_expr},"
#         f"pad=iw:ih+40:0:40:blue,fps=25[h];"
#         f"[h]drawtext=font='monospace':text='{text}':"
#         f"fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:"
#         f"x=(w-text_w)/2:y=10\" -t 00:00:07 \"{output_file}\""
#     )

#     print(f"[INFO] Running FFmpeg command:\n{cmd}\n")
#     os.system(cmd)
import argparse
import os
import sys
import subprocess

def convert_gif_to_mp4(gif_path, mp4_path):
    cmd = [
        'ffmpeg', '-y', '-i', gif_path,
        '-movflags', 'faststart', '-pix_fmt', 'yuv420p',
        '-vf', 'fps=25,scale=iw:ih:flags=lanczos',
        mp4_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_ffmpeg_command(cmd):
    # print(f"[INFO] Running command:\n{cmd}\n")
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_dir', type=str, help='Directory where GIFs are stored.')
    parser.add_argument('--layout', default='linear', help='Which layout to use (linear or grid)')
    parser.add_argument('--how_many', type=int, default=10, help='Number of GIFs to combine')
    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        # print(f"[ERROR] Directory does not exist: {args.video_dir}")
        sys.exit(1)

    # print(f"[INFO] Reading from directory: {args.video_dir}")
    all_files = os.listdir(args.video_dir)

    gif_files = [f for f in all_files if f.endswith('.gif')]
    gif_files.sort()

    how_many_vids = min(len(gif_files), args.how_many)
    gif_files = gif_files[:how_many_vids]

    # Convert all GIFs to MP4
    mp4_files = []
    for gif in gif_files:
        gif_path = os.path.join(args.video_dir, gif)
        mp4_path = gif_path.replace('.gif', '.mp4')
        convert_gif_to_mp4(gif_path, mp4_path)
        mp4_files.append(mp4_path)

    # Layout for combining
    if args.layout == 'linear':
        layout = ["0_0"] + ['+'.join([f'w{i}' for i in range(j)])+'_0' for j in range(1, how_many_vids)]
    elif args.layout == 'grid':
        layout = ["0_0", "w0_0", "w0+w1_0", "w0+w1+w2_0", 
                  "0_h0", "w4_h0", "w4+w5_h0", "w4+w5+w6_h0", 
                  "0_h0+h4", "w8_h0+h4", "w8+w9_h0+h4", "w8+w9+w10_h0+h4"]

    # Read description
    desc_path = os.path.join(args.video_dir, 'desc.txt')
    if os.path.exists(desc_path):
        with open(desc_path, 'r') as f:
            lines = f.read().splitlines()
            desc = lines[0] if len(lines) > 0 else "No description"
            rank = lines[1] if len(lines) > 1 else "N/A"
    else:
        desc, rank = "No description available", "N/A"

    text = f"{desc} (Rank of correct result = {rank})"

    input_args = " ".join([f"-i \"{v}\"" for v in mp4_files])
    input_streams = "".join([f"[{i}:v]" for i in range(how_many_vids)])
    layout_expr = "|".join(layout[:how_many_vids])
    combined_video = os.path.join(args.video_dir, "combined.mp4")

    # FFmpeg command to combine MP4s into one video
    combine_cmd = (
        f"ffmpeg {input_args} -y -filter_complex "
        f"\"{input_streams}xstack=inputs={how_many_vids}:layout={layout_expr},"
        f"pad=iw:ih+50:0:40:blue,drawtext=font='monospace':text='{text}':"
        f"fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:"
        f"x=(w-text_w)/2:y=10\" "
        f"-c:v libx264 -crf 18 -preset veryslow \"{combined_video}\""
    )
    run_ffmpeg_command(combine_cmd)

    # Generate palette
    palette_path = os.path.join(args.video_dir, "palette.png")
    run_ffmpeg_command(
        f"ffmpeg -y -i \"{combined_video}\" -vf \"fps=25,scale=800:-1:flags=lanczos,palettegen\" \"{palette_path}\""
    )

    # Generate final GIF
    final_gif_path = os.path.join(args.video_dir, "output.gif")
    run_ffmpeg_command(
        f"ffmpeg -y -i \"{combined_video}\" -i \"{palette_path}\" "
        f"-filter_complex \"fps=25,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse\" \"{final_gif_path}\""
    )

    # print(f"[INFO] High-quality GIF saved to: {final_gif_path}")
