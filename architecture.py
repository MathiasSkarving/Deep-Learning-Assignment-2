import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_cnn_vertical(save_path="cnn_architecture.png"):
    # (Stage/Type, Layer Name, Details, Output Shape, Box Color)
    pipeline = [
        ("INPUT",  "Input Image",      "3-channel RGB",               "3 × 28 × 28",   "#ECEFF1"),
        ("POOL",   "AvgPool2d",        "k=3, s=1, p=1",               "3 × 28 × 28",   "#FFF3E0"),
        ("CONV",   "Conv Block 1",     "Conv2d (5×5) + BN + ReLU",    "32 × 28 × 28",  "#E8F5E9"),
        ("CONV",   "Conv Block 2",     "Conv2d (3×3) + BN + ReLU",    "64 × 28 × 28",  "#E8F5E9"),
        ("POOL",   "MaxPool2d",        "k=2, s=2",                    "64 × 14 × 14",  "#FFF3E0"),
        ("CONV",   "Conv Block 3",     "Conv2d (3×3) + BN + ReLU",    "128 × 14 × 14", "#E8F5E9"),
        ("POOL",   "MaxPool2d",        "k=2, s=2",                    "128 × 7 × 7",   "#FFF3E0"),
        ("FLAT",   "Flatten",          "x.reshape(x.shape[0], -1)",   "6,272",         "#EDE7F6"),
        ("DENSE",  "Dense Block 1",    "Linear (512) + ReLU + Dropout(0.3)", "512",    "#EDE7F6"),
        ("DENSE",  "Dense Block 2",    "Linear (256) + ReLU + Dropout(0.3)", "256",    "#EDE7F6"),
        ("OUT",    "Classifier",       "Linear (num_classes)",        "8",             "#FCE4EC"),
    ]

    fig, ax = plt.subplots(figsize=(9, 11), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, len(pipeline) * 1.1 + 0.5)
    ax.axis("off")

    card_w = 8.6
    card_h = 0.68
    x_left = 0.7
    y_start = len(pipeline) * 1.1 - 0.5

    for i, (badge, name, details, out_shape, bg_col) in enumerate(pipeline):
        y = y_start - (i * 1.1)

        # 1. Main Card Container
        card = patches.FancyBboxPatch(
            (x_left, y), card_w, card_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=bg_col, edgecolor="#90A4AE", linewidth=1.2
        )
        ax.add_patch(card)

        # 2. Type Badge (Left pill)
        badge_box = patches.FancyBboxPatch(
            (x_left + 0.15, y + 0.12), 1.1, card_h - 0.24,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#37474F", edgecolor="none"
        )
        ax.add_patch(badge_box)
        ax.text(x_left + 0.7, y + card_h / 2, badge, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white")

        # 3. Layer Name & Details (Middle)
        ax.text(x_left + 1.45, y + card_h * 0.62, name, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color="#212121")
        ax.text(x_left + 1.45, y + card_h * 0.25, details, ha="left", va="center",
                fontsize=8, color="#546E7A")

        # 4. Output Shape Badge (Right pill)
        shape_box = patches.FancyBboxPatch(
            (x_left + card_w - 2.3, y + 0.12), 2.15, card_h - 0.24,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#FFFFFF", edgecolor="#CFD8DC", linewidth=1.0
        )
        ax.add_patch(shape_box)
        ax.text(x_left + card_w - 1.22, y + card_h / 2, out_shape, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#1E88E5", family="monospace")

        # 5. Connector Arrow
        if i < len(pipeline) - 1:
            ax.annotate(
                "", xy=(x_left + card_w / 2, y - 0.38),
                xytext=(x_left + card_w / 2, y),
                arrowprops=dict(arrowstyle="->", color="#78909C", lw=1.5, mutation_scale=12)
            )

    plt.title("Custom CNN Architecture for BloodMNIST", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

draw_cnn_vertical()