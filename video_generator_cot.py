import pygame
import os
import random
import imageio
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
import time
from openai import OpenAI


class SyntheticDatasetLoader:
    PROMPT_FORMAT = """
        I will provide you with an image, an original question, and its answer related to the image. 
        Your task is to rewrite the question in such a way that answering it requires step-by-step Chain-of-Thought (CoT) reasoning with numerical or mathematical expressions where applicable.
        The reasoning process should start with a goal aknowledging expression like: "The goal is", "So, the objective is". Then, inlcude problem solving expressions like: "Let me think this matter step by step", "let me think", "a way to solve it is", "oh, I see it", "mmm, interesting", "this is probaby correct" or other natural language thought expressions.

        Please strictly do not include \"Answer:\" in the question part to avoid confusion and leakage.

        Input Format:
            Original Question: {original_question}
            Original Answer: {original_answer}

        Output Format:
            Answer: [answer with reasoning steps, including calculations where applicable]
                    <think>step-by-step reasoning process</think>
                    <answer>easy to verify answer</answer>
    """

    def __init__(self, output_dir="dataset", screen_size=(64, 64), video_length=30):
        base_path = Path(__file__).parent.resolve()
        self.output_dir = (base_path / output_dir).resolve()
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

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        print(f"\U0001F4C2 Dataset will be saved to: {self.output_dir}")

    def generate_dataset(self, num_samples=10):
        metadata = []

        samples_per_class = num_samples // len(self.classes)
        extra = num_samples % len(self.classes)
        class_list = self.classes * samples_per_class + random.sample(self.classes, extra)
        random.shuffle(class_list)

        for idx, motion_type in enumerate(class_list):
            video_path = self.video_dir / f"{idx:03d}.mp4"
            self._generate_video(motion_type, video_path)

            question = "<video>\nIn which direction is the ball moving?\nOptions:\n(A) Left to Right\n(B) Right to Left\n(C) Falling Down\n(D) Ascending"
            answer = self.option_labels[motion_type]

            composite_image = self._create_motion_composite(video_path)
            cot_response = self._generate_cot_response(composite_image, question, answer)

            conversations = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": f"Answer: {answer}"}
            ]

            if cot_response:
                conversations.append({"from": "cot_gpt", "value": cot_response})

            metadata.append({
                "video": str(video_path.resolve()),
                "conversations": conversations
            })

        with open(self.meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\u2705 Dataset generated with {num_samples} samples.")
        print(f"\U0001F5C2\ufe0f  Videos saved in: {self.video_dir.resolve()}")
        print(f"\ud83d\udcdd Metadata saved in: {self.meta_file.resolve()}")

    def _generate_video(self, motion_type, save_path):
        pygame.init()
        screen = pygame.Surface((self.screen_width, self.screen_height))

        black = (0, 0, 0)
        white = (255, 255, 255)
        radius = 20

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
            frame = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
            frames.append(frame)

        imageio.mimsave(save_path, frames, fps=30)
        pygame.quit()

    def _create_motion_composite(self, video_path, sample_indices=[0, 20, 40, 59]):
        reader = imageio.get_reader(video_path)
        frames = [Image.fromarray(reader.get_data(i)).convert("RGBA") for i in sample_indices]
        base = frames[0].copy()
        for f in frames[1:]:
            base = Image.blend(base, f, alpha=0.5)
        return base.convert("RGB")

    def _generate_cot_response(self, image: Image.Image, question: str, answer: str, max_retries=5):
        def image_to_base64(img: Image.Image):
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="JPEG")
            return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

        prompt = self.PROMPT_FORMAT.format(original_question=question, original_answer=answer)
        data_url = image_to_base64(image)

        messages = [
            {"role": "system", "content": "You are an expert to analyze the image and provide a Chain of Though CoT answer to the user."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt}
            ]}
        ]

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"\u26a0\ufe0f Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2 ** attempt + random.uniform(0, 1))

        print("GPT failed after retries.")
        return None


if __name__ == "__main__":
    loader = SyntheticDatasetLoader(output_dir="qwen-vl-finetune/qwenvl/data/synthetic_datasets/my_ball_dataset")
    loader.generate_dataset(num_samples=20)
