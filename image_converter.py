import sys
from PIL import Image

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
        print("  Convert to PPM  : python3 image_converter.py to_ppm <input.png/jpg> <output.ppm>")
        print("  Convert from PPM: python3 image_converter.py to_png <input.ppm> <output.png>")
        sys.exit(1)
        
    mode = sys.argv[1].lower()
    in_file = sys.argv[2]
    out_file = sys.argv[3]
    
    if mode == 'to_ppm':
        to_ppm(in_file, out_file)
    elif mode == 'to_png' or mode == 'to_jpg':
        to_image(in_file, out_file)
    else:
        print(f"Unknown mode: {mode}. Use 'to_ppm' or 'to_png'.")
        sys.exit(1)
