import os
from PIL import Image, ImageDraw

def create_examples():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(dir_path, "few_shot_examples")
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Undercut
    img1 = Image.new("RGB", (400, 100), color=(120, 120, 120))
    draw1 = ImageDraw.Draw(img1)
    # Draw weld bead
    draw1.rectangle([0, 30, 400, 70], fill=(90, 95, 100))
    # Draw undercut defect: [ymin, xmin, ymax, xmax] -> [350, 150, 470, 250] in 0-1000 scale.
    # On a 400x100 image, y coordinates are multiplied by 100/1000 = 0.1, x coordinates by 400/1000 = 0.4.
    # ymin=35, xmin=60, ymax=47, xmax=100.
    draw1.rectangle([60, 35, 100, 47], fill=(255, 50, 50))
    img1.save(os.path.join(target_dir, "example_undercut.jpg"), "JPEG")
    
    # 2. Porosity
    img2 = Image.new("RGB", (400, 100), color=(120, 120, 120))
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([0, 30, 400, 70], fill=(90, 95, 100))
    # Draw porosity: [ymin, xmin, ymax, xmax] -> [450, 480, 530, 520] in 0-1000 scale.
    # ymin=45, xmin=192, ymax=53, xmax=208.
    draw2.ellipse([192, 45, 208, 53], fill=(0, 224, 255))
    img2.save(os.path.join(target_dir, "example_porosity.jpg"), "JPEG")
    
    # 3. Reinforcement
    img3 = Image.new("RGB", (400, 100), color=(120, 120, 120))
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([0, 30, 400, 70], fill=(90, 95, 100))
    # Draw excess reinforcement: [ymin, xmin, ymax, xmax] -> [380, 600, 580, 800] in 0-1000 scale.
    # ymin=38, xmin=240, ymax=58, xmax=320.
    draw3.rectangle([240, 38, 320, 58], fill=(255, 45, 235))
    img3.save(os.path.join(target_dir, "example_reinforcement.jpg"), "JPEG")
    
    print("Example images generated successfully!")

if __name__ == "__main__":
    create_examples()
