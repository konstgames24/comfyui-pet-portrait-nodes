# Pet Portrait Layout — ComfyUI nodes

Nodes for the "photo → pastel painted pet portrait" workflow. Pure PyTorch, no extra dependencies.

Install: put this folder into `ComfyUI/custom_nodes/` (or `git clone` the repo there) and restart ComfyUI.

| Node | What it does |
|---|---|
| Pet Pastel Background | Solid pastel canvas (8 tones) plus its name for the prompt |
| Pet Pastel Palette | 24 pastel tones; pick one, random by group, or auto to suit the coat |
| Pet Portrait Layout | Places the subject centred with empty space above, chest past the bottom edge |
| Pet Portrait Layout (padded) | Same, but with space below and a brush-like fade at the bottom of the bust |
| Pet Group Layout | Several animals side by side on one image, or one portrait each |
| Pet Split Animals | Splits a photo into one crop per animal from Qwen3-VL `bbox_2d` boxes (list output) |
| Pet Head Portrait Layout | Frames an untouched photo by the animal's head box |
| Pet Face Mask | Per-pixel strength mask: weaker painterly pass on the face (with Differential Diffusion) |
| Pet Restore Colors | Brushwork from a painting, colours from the photo |
| Pet Flatten Background | Exact flat background while keeping painted fur wisps at the outline |
| Pet Color Palette | Gradient-map recolouring of the animal (not used by the workflows) |
