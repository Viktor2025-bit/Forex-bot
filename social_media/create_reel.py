import os
from moviepy.editor import ImageClip, concatenate_videoclips, CompositeVideoClip, ColorClip

def create_reel(image_folder, output_file="output_reel.mp4", duration_per_image=3.0):
    """
    Creates a 9:16 social media reel from a folder of images.
    """
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort() # Ensure consistent order (e.g. expectation then reality)
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return

    clips = []
    
    # Instagram Reel / TikTok resolution
    W, H = 1080, 1920 
    
    print(f"Found {len(image_files)} images. Processing...")

    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        
        # Create an ImageClip
        img_clip = ImageClip(img_path).set_duration(duration_per_image)
        
        # Resize to fit width (1080) while maintaining aspect ratio
        img_clip = img_clip.resize(width=W)
        
        # If the image is shorter than 1920, center it on a black background
        if img_clip.h < H:
            # Create a background (can be blurred version of image or solid color)
            # For simplicity, let's use a blurred background style often seen on Reels
            
            # 1. Background: Resize original to fill height (zoomed in/cropped) and blur
            bg_clip = ImageClip(img_path).set_duration(duration_per_image)
            bg_clip = bg_clip.resize(height=H)
            bg_clip = bg_clip.crop(x1=bg_clip.w/2 - W/2, width=W, height=H) # Center crop
            # Note: Gaussian blur in moviepy can be slow, using a dark gray bg is safer for speed
            # bg_clip = bg_clip.filter(GaussianBlur(10)) 
            
            # Alternative: Solid color background
            bg_clip = ColorClip(size=(W, H), color=(20, 20, 20), duration=duration_per_image)
            
            # Composite: Put the resized image on top of the background, centered
            final_clip = CompositeVideoClip([bg_clip, img_clip.set_position("center")])
        else:
            # If image is taller/large, just crop center
            final_clip = img_clip.crop(x1=img_clip.w/2 - W/2, width=W, height=H)

        clips.append(final_clip)

    # Concatenate all clips
    # method='compose' is safer for different sizes, but we standardized them above
    final_video = concatenate_videoclips(clips, method="compose")
    
    # Write output
    print(f"Writing video to {output_file}...")
    final_video.write_videofile(output_file, fps=24)
    print("Done!")

if __name__ == "__main__":
    # Default to the 'assets' folder in the same directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")
    
    create_reel(assets_dir)
