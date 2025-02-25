import os

# Configuration
IS_HF_SPACES = os.getenv("SYSTEM", "") == "spaces"
MAX_CHARACTERS = 2000

HEADER_MARKDOWN = """
# Zonos v0.1
State of the art text-to-speech model [[model]](https://huggingface.co/collections/Zyphra/zonos-v01-67ac661c85e1898670823b4f). [[blog]](https://www.zyphra.com/post/beta-release-of-zonos-v0-1), [[Zyphra Audio (hosted service)]](https://maia.zyphra.com/sign-in?redirect_url=https%3A%2F%2Fmaia.zyphra.com%2Faudio)
## Unleashed
Use this space to generate long-form speech up to around ~2 minutes in length. To generate an unlimited length, clone this space and run it locally.
### Tips
- When providing prefix audio, include the text of the prefix audio in your speech text to ensure a smooth transition.
- The appropriate range of Speaking Rate and Pitch STD are highly dependent on the speaker audio. Start with the defaults and adjust as needed.
- Emotion sliders do not completely function intuitively, and require some experimentation to get the desired effect.
""".strip()