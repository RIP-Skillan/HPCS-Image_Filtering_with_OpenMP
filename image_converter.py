import sys
from PIL import Image
import os

def to_ppm(input_path, output_path):
    try:
        img = Image.open(input_path).convert('RGB')
        img.save(output_path, format='PPM')
        print(f"Successfully converted '{input_path}' to PPM '{output_path}' ({img.width}x{img.height})")
    except Exception as e:
        print(f"Error converting to PPM: {e}")

def to_image(input_path, output_path):
    try:
        img = Image.open(input_path)
        img.save(output_path)
        print(f"Successfully converted PPM '{input_path}' to '{output_path}'")
    except Exception as e:
        print(f"Error converting from PPM: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage:")
        print("  Convert to PPM  : python3 image_converter.py to_ppm <input_folder> <output_folder>")
        print("  Convert from PPM: python3 image_converter.py to_png <input_folder> <output_folder>")
        sys.exit(1)
    

    mode = sys.argv[1].lower()
    in_folder = sys.argv[2]
    out_folder = sys.argv[3]
    #in_file = sys.argv[2]
    #out_file = sys.argv[3]
    files = [f for f in os.listdir(in_folder) if os.path.isfile(in_folder + '/' + f)] 
    if mode == 'to_ppm':
        for file in files:
            to_ppm(os.path.join(in_folder, file), os.path.join(out_folder, file[:-3]+'ppm'))
    elif mode == 'to_png' or mode == 'to_jpg':
        for file in files:
            to_image(os.path.join(in_folder, file), os.path.join(out_folder, file))
    else:
        print(f"Unknown mode: {mode}. Use 'to_ppm' or 'to_png'.")
        sys.exit(1)
