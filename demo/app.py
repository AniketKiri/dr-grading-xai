"""Optional local demo: grade one fundus photograph with the trained CORN model. Research prototype.

    python demo/app.py            # then open http://127.0.0.1:7860

Needs runs/resnet50_corn/checkpoints/best.pt (not in the repository) and, for the six examples, the
MMRDR CFP images under data/. Every step reuses the pipeline's own functions (see pipeline.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import Pipeline, example_ids, original_path  # noqa: E402

from drxai.plots import SHORT  # noqa: E402

BANNER = ('<div class="banner">Research prototype. Not a clinically validated diagnostic tool.</div>')
CAPTION = "Regions that raise the predicted severity. A plausible location does not prove a lesion was recognised."
LOW_SHARP = ("Low sharpness: on this model, blur raised confidence while lowering accuracy (report Section 3.3). "
             "Treat the prediction with caution.")
PLAIN = ["no DR", "mild NPDR", "moderate NPDR", "severe NPDR", "proliferative DR"]
PREDICTED = "#4B2FB5"
OTHER = "#C9C3E6"

# Badges always carry their text, so the colour is never the only signal.
CSS = """
.banner {background:#7a1f1f; color:#fff; font-weight:700; font-size:1.15em; padding:10px 14px;
         border-radius:6px; text-align:center;}
.badge {display:inline-block; padding:4px 10px; border-radius:12px; font-weight:600; margin:2px 6px 2px 0;
        border:1px solid rgba(0,0,0,0.25);}
.badge.green {background:#d6f0dc; color:#14532d;}
.badge.amber {background:#fde9c4; color:#78350f;}
.badge.red {background:#fbd5d5; color:#7f1d1d;}
.summary {font-size:1.1em; font-weight:600; margin:4px 0 8px 0;}
"""


def badge(text: str, colour: str) -> str:
    return f'<span class="badge {colour}">{text}</span>'


def prob_figure(probs, pred: int, K: int):
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.barh(range(K), probs, color=[PREDICTED if k == pred else OTHER for k in range(K)])
    for k, v in enumerate(probs):
        # labels always sit just past the bar end; the x range leaves room even for a full bar
        ax.text(v + 0.02, k, f"{v:.2f}", va="center", ha="left", fontsize=8,
                fontweight="bold" if k == pred else "normal")
    ax.set_yticks(range(K), SHORT[:K], fontsize=8)
    for k, t in enumerate(ax.get_yticklabels()):
        t.set_fontweight("bold" if k == pred else "normal")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Calibrated probability", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def cam_figure(img, cam, alpha: float):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].imshow(img)
    axes[0].set_title("Network input", fontsize=9)
    axes[1].imshow(img)
    axes[1].imshow(cam, cmap="jet", alpha=float(alpha), vmin=0, vmax=1)
    axes[1].set_title(f"Grad-CAM (target: expected grade), opacity {alpha:.2f}", fontsize=9)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    return fig


def build(pipe: Pipeline) -> gr.Blocks:
    pred = pd.read_csv(pipe.run_dir / "predictions_test.csv", dtype={"image_id": str})
    truth = dict(zip(pred["image_id"], pred["grade"]))
    ids = example_ids(pred)
    labels = {f"{i}  (reference grade {SHORT[int(truth[i])]})" + ("  confident miss" if i == "ts000729" else ""): i
              for i in ids}

    def run(upload, choice, alpha):
        if upload is not None:
            img, ref_line = Image.open(upload), "Reference grade: not available for an uploaded image."
        elif choice:
            i = labels[choice]
            img = Image.open(original_path(pipe.cfg, i))
            ref_line = f"Reference grade (MMRDR label): {SHORT[int(truth[i])]}"
        else:
            raise gr.Error("Upload a fundus photograph or choose an example.")
        r = pipe(img)
        confident = r.action == "confident: no uncertainty flag"
        badges = badge(r.action, "green" if confident else "amber")
        if r.low_sharpness:
            badges += badge("image quality warning", "red")
        # the rule reports uncertainty, not what to do with the eye, so the summary states the flag only
        flag = "No uncertainty flag." if confident else "Uncertain: flag for extended optometrist review."
        summary = (f'<div class="summary">Likely {PLAIN[r.pred]} (confidence {r.confidence:.2f}). '
                   f"Probability of referable DR {r.p_referable:.2f}. {flag}</div>")
        details = (f"- Predicted grade: **{SHORT[r.pred]}**\n"
                   f"- Calibrated confidence: **{r.confidence:.2f}** (temperature {pipe.T:.3f}, fitted on validation)\n"
                   f"- P(referable DR, grade >= {pipe.ref}) = **{r.p_referable:.2f}**\n"
                   f"- Uncertainty rule: no flag if confidence >= {pipe.tau:.3f} (threshold set on validation)")
        quality = (f"Image quality check (experimental, not validated): sharpness {r.sharpness:.1f}, "
                   f"low band edge {pipe.sharp_low:.1f} (training images).")
        if r.low_sharpness:
            quality = f"**{LOW_SHARP}**\n\n" + quality
        if r.grey_surround:
            quality += " Grey camera surround detected and set to black, as in training."
        judgement = f"### Decision remains with the optometrist.\n{ref_line}"
        return (summary + badges, details, prob_figure(r.probs, r.pred, pipe.K), quality,
                cam_figure(r.display, r.cam, alpha), (r.display, r.cam), judgement)

    def redraw(state, alpha):
        # opacity changes only the drawing, so the model is not run again
        if state is None:
            return gr.update()
        return cam_figure(state[0], state[1], alpha)

    with gr.Blocks(title="DR grading demo") as demo:
        gr.HTML(BANNER)
        gr.Markdown("Five-grade diabetic retinopathy grading (ICDR 0 to 4), ResNet-50 with an ordinal (CORN) head "
                    f"trained on MMRDR. Running on **{pipe.device.upper()}**.")
        state = gr.State(None)
        with gr.Row(equal_height=False):
            # min_width makes the two columns stack on narrow screens
            with gr.Column(scale=6, min_width=420):
                with gr.Row():
                    upload = gr.Image(type="filepath", label="Upload a colour fundus photograph", height=220)
                    with gr.Column():
                        choice = gr.Dropdown(list(labels), label="Or choose a test example (lowest test id per "
                                             "grade, plus a confident miss)", value=None)
                        go = gr.Button("Grade image", variant="primary")
                cam = gr.Plot(label="Image and Grad-CAM", show_label=False)
                alpha = gr.Slider(0, 1, value=0.5, step=0.05, label="Grad-CAM opacity",
                                  info="Lower it to see the retina under the heatmap")
                quality = gr.Markdown()
            with gr.Column(scale=5, min_width=360):
                gr.Markdown("## 1  MODEL PREDICTION")
                head = gr.HTML()
                prediction = gr.Markdown()
                bars = gr.Plot(label="Grade distribution")
                gr.Markdown("## 2  EXPLANATION")
                gr.Markdown(CAPTION + " The heatmap is shown next to the input image; use the opacity slider to see the retina beneath it.")
                gr.Markdown("## 3  CLINICAL JUDGEMENT")
                judgement = gr.Markdown("### Decision remains with the optometrist.")
        # a new upload clears the example choice and vice versa, so the panels always describe one image
        upload.upload(lambda: None, outputs=choice)
        choice.change(lambda c: None if c else gr.update(), inputs=choice, outputs=upload)
        go.click(run, inputs=[upload, choice, alpha],
                 outputs=[head, prediction, bars, quality, cam, state, judgement])
        alpha.change(redraw, inputs=[state, alpha], outputs=cam)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/resnet50_corn")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build(Pipeline(args.run)).launch(server_name="127.0.0.1", server_port=args.port, css=CSS)


if __name__ == "__main__":
    main()
