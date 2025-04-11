import pygame
import os
import random
import imageio
import json
from pathlib import Path


class SyntheticDatasetLoader:
    def __init__(self, output_dir="dataset", screen_size=(224, 224), video_length=60):
        self.output_dir = Path(output_dir)
        self.video_dir = self.output_dir / "videos"
        self.meta_file = self.output_dir / "metadata.json"
        self.screen_width, self.screen_height = screen_size
        self.video_length = video_length
        self.classes = ["left_to_right", "right_to_left", "falling", "ascending"]
        self.option_labels = {
            "left_to_right": "(A) Left to Right",
            "right_to_left": "(B) Right to Left",
            "falling": "(C) Falling Down",
            "ascending": "(D) Ascending"
        }

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset(self, num_samples=10):
        metadata = []

        # Evenly distribute classes
        samples_per_class = num_samples // len(self.classes)
        extra = num_samples % len(self.classes)
        class_list = self.classes * samples_per_class + random.sample(self.classes, extra)
        random.shuffle(class_list)

        for idx, motion_type in enumerate(class_list):
            video_path = self.video_dir / f"{idx:03d}.mp4"
            self._generate_video(motion_type, video_path)

            entry = {
                "video": str(video_path.resolve()),  # ✅ Absolute path here
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\nIn which direction is the ball moving?\nOptions:\n(A) Left to Right\n(B) Right to Left\n(C) Falling Down\n(D) Ascending"
                    },
                    {
                        "from": "gpt",
                        "value": f"Answer: {self.option_labels[motion_type]}"
                    }
                ]
            }
            metadata.append(entry)

        with open(self.meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Dataset generated with {num_samples} samples.")
        print(f"🗂️  Videos saved in: {self.video_dir.resolve()}")
        print(f"📝 Metadata saved in: {self.meta_file.resolve()}")

    def _generate_video(self, motion_type, save_path):
        pygame.init()
        screen = pygame.Surface((self.screen_width, self.screen_height))

        black = (0, 0, 0)
        white = (255, 255, 255)
        radius = 20

        # Start positions and velocities
        if motion_type == "left_to_right":
            x, y = -radius, self.screen_height // 2
            dx, dy = 5, 0
        elif motion_type == "right_to_left":
            x, y = self.screen_width + radius, self.screen_height // 2
            dx, dy = -5, 0
        elif motion_type == "falling":
            x, y = self.screen_width // 2, -radius
            dx, dy = 0, 5
        elif motion_type == "ascending":
            x, y = self.screen_width // 2, self.screen_height + radius
            dx, dy = 0, -5

        frames = []
        for _ in range(self.video_length):
            x += dx
            y += dy

            screen.fill(black)
            pygame.draw.circle(screen, white, (int(x), int(y)), radius)
            frame = pygame.surfarray.array3d(screen)
            frame = frame.transpose([1, 0, 2])
            frames.append(frame)

        imageio.mimsave(save_path, frames, fps=30)
        pygame.quit()

if __name__ == "__main__":
    loader = SyntheticDatasetLoader(output_dir="qwen-vl-finetune/qwenvl/data/synthetic_datasets/my_ball_dataset")
    loader.generate_dataset(num_samples=20)
