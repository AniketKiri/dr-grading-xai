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

BANNER = "**Research prototype. Not a clinically validated diagnostic tool.**"
CAPTION = "Regions that raise the predicted severity. A plausible location does not prove a lesion was recognised."
LOW_SHARP = ("**Low sharpness:** on this model, blur raised confidence while lowering accuracy (report Section 3.3). "
             "Treat the prediction with caution.")


def prob_figure(probs, pred: int, K: int):
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.barh(range(K), probs, color=["#4B2FB5" if k == pred else "#C9C3E6" for k in range(K)])
    for k, v in enumerate(probs):
        ax.text(min(v + 0.02, 0.9), k, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(K), SHORT[:K], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Calibrated probability", fontsize=8)
    fig.tight_layout()
    return fig


def cam_figure(img, cam):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].imshow(img)
    axes[0].set_title("Network input", fontsize=9)
    axes[1].imshow(img)
    axes[1].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM (target: expected grade)", fontsize=9)
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

    def run(upload, choice):
        if upload is not None:
            img, ref_line = Image.open(upload), "Reference grade: not available for an uploaded image."
        elif choice:
            i = labels[choice]
            img, ref_line = Image.open(original_path(pipe.cfg, i)), f"Reference grade (MMRDR label for {i}): {SHORT[int(truth[i])]}."
        else:
            raise gr.Error("Upload a fundus photograph or choose an example.")
        r = pipe(img)
        quality = (f"Image quality check (experimental, not validated): sharpness {r.sharpness:.1f}, "
                   f"low band edge {pipe.sharp_low:.1f} (training images).")
        if r.low_sharpness:
            quality = LOW_SHARP + "\n\n" + quality
        if r.grey_surround:
            quality += " Grey camera surround detected and set to black, as in training."
        prediction = (f"### {SHORT[r.pred]}\n"
                      f"- Calibrated confidence: **{r.confidence:.2f}** (temperature {pipe.T:.3f}, fitted on validation)\n"
                      f"- P(referable DR, grade >= {pipe.ref}) = **{r.p_referable:.2f}**\n"
                      f"- Action under the referral rule (threshold {pipe.tau:.3f}, set on validation): "
                      f"**{r.action}**")
        judgement = f"### Decision remains with the optometrist.\n{ref_line}"
        return (quality, prediction, prob_figure(r.probs, r.pred, pipe.K), cam_figure(r.display, r.cam), judgement)

    with gr.Blocks(title="DR grading demo") as demo:
        gr.Markdown(BANNER)
        gr.Markdown("Five-grade diabetic retinopathy grading (ICDR 0 to 4), ResNet-50 with an ordinal (CORN) head "
                    f"trained on MMRDR. Running on **{pipe.device.upper()}**.")
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.Image(type="filepath", label="Upload a colour fundus photograph", height=260)
                choice = gr.Dropdown(list(labels), label="Or choose a test example (lowest test id per grade, "
                                     "plus a confident miss)", value=None)
                go = gr.Button("Grade image", variant="primary")
                quality = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("## 1  MODEL PREDICTION")
                prediction = gr.Markdown()
                bars = gr.Plot(label="Grade distribution")
        gr.Markdown("## 2  EXPLANATION")
        cam = gr.Plot(label="Grad-CAM")
        gr.Markdown(CAPTION)
        gr.Markdown("## 3  CLINICAL JUDGEMENT")
        judgement = gr.Markdown("### Decision remains with the optometrist.")
        # a new upload clears the example choice and vice versa, so the panel always shows one image
        upload.upload(lambda: None, outputs=choice)
        choice.change(lambda c: None if c else gr.update(), inputs=choice, outputs=upload)
        go.click(run, inputs=[upload, choice], outputs=[quality, prediction, bars, cam, judgement])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/resnet50_corn")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build(Pipeline(args.run)).launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
