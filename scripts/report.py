"""Plot defensive trade-offs and generate a measured technical report."""

import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT=Path("results")
POLICIES={
    "wait":"Wait only","alert_rule":"Alert rule","inspect_rule":"Inspect rule","oracle":"Oracle",
    "dqn":"DQN","ppo":"PPO","ppo_security":"Security PPO","ppo_availability":"Availability PPO",
    "ppo_no_inspection":"No-inspection PPO","ppo_masked":"Masked PPO",
    "ppo_masked_prior":"Prior + masked PPO","imitation_only":"Imitation only",
}
SCENARIOS={"iid":"Independent","bursty":"Burst","shifted":"Shifted"}
COLORS=["#8197ad","#995d40","#327d70","#5d7698","#ac6e34","#b76b79","#826290",
        "#adb151","#9e6c5d","#3c7c9e","#00756e","#364960"]


def figures(data,index):
    fig,axes=plt.subplots(1,3,figsize=(12,3.55),sharey=True)
    chosen=("wait","alert_rule","inspect_rule","dqn","ppo","ppo_masked","ppo_masked_prior","imitation_only")
    colors={m:COLORS[i] for i,m in enumerate(POLICIES)}
    for scenario,ax in zip(SCENARIOS,axes,strict=True):
        for name in chosen:
            stats=index[scenario,name]
            ax.scatter(100*(1-stats["outage"]["mean"]),100*stats["exposure"]["mean"],
                       color=colors[name],label=POLICIES[name],s=48,alpha=.9)
        ax.set(xlabel="Availability (%)",title=SCENARIOS[scenario])
        ax.grid(alpha=.23)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Uncontained exposure (%)")
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc="upper center",frameon=False)
    fig.tight_layout(rect=(0,0,1,.83))
    fig.savefig(ROOT/"tradeoffs.png",dpi=170,facecolor="white",bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,3.5))
    for model in ("dqn","ppo","ppo_masked","ppo_masked_prior"):
        rows=[x["history"] for x in data["training"] if x["model"]==model]
        xs=[x["steps"] for x in rows[0]]
        ys=np.asarray([[x["mean_reward"] for x in row] for row in rows])
        ax.plot(xs,ys.mean(0),"o-",color=colors[model],label=POLICIES[model],ms=4)
        ax.fill_between(xs,ys.min(0),ys.max(0),color=colors[model],alpha=.12)
    ax.set(xlabel="Training interactions",ylabel="Validation episode reward")
    ax.legend(frameon=False,ncol=2)
    ax.grid(alpha=.23)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT/"learning.png",dpi=170,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def pdf(data,index):
    dark=colors.HexColor("#193b4e")
    fonts=Path(matplotlib.get_data_path())/"fonts"/"ttf"
    pdfmetrics.registerFont(TTFont("StudySans",str(fonts/"DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("StudySans-Bold",str(fonts/"DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("StudySans",normal="StudySans",bold="StudySans-Bold")
    styles=getSampleStyleSheet()
    for k in ("Normal","BodyText"):
        styles[k].fontName="StudySans"
    for k in ("Title","Heading2"):
        styles[k].fontName="StudySans-Bold"
        styles[k].textColor=dark
    styles["Title"].alignment=0
    styles["Title"].fontSize=22
    styles["Title"].leading=27
    styles["BodyText"].leading=14
    styles["BodyText"].spaceAfter=8
    body=[]
    def p(s,k="BodyText"):
        body.append(Paragraph(s,styles[k]))
    def section(t,s):
        p(t,"Heading2")
        p(s)
    def graphic(asset,height):
        body.append(Image(str(ROOT/asset),width=483,height=height,kind="proportional"))
    def metric(arm,key,scenario="shifted"):
        return index[scenario,arm][key]["mean"]
    def reward(arm,scenario="shifted"):
        return f"{metric(arm,'reward',scenario):.2f}"

    p("Reinforcement Learning for Defensive Incident Response","Title")
    p("A paired simulation study of exposure, availability and response cost")
    p(f"<b>Abstract.</b> Across {len(data['rows']):,} policy-episode measurements, "
      "we compare DQN, PPO, handcrafted rules and a latent-state reference on paired "
      "simulated incidents. The shifted-cohort mean balanced return is "
      f"{reward('ppo')} for PPO, {reward('inspect_rule')} for the inspection rule, "
      f"and {reward('ppo_masked_prior')} for rule-initialized masked PPO. "
      "Observed-state action constraints and imitation are evaluated as independent interventions.")
    section("1. Research question",
            "Can learned defenders lower incident exposure without unacceptable availability costs? "
            "We test whether state-based action masking, noisy inspection signals and objective "
            "weights account for policy differences.")
    section("2. Related work",
            "DQN uses approximate action values [1] and PPO constrains policy changes [2]. "
            "Defensive simulation environments motivate repeatable transition and reward "
            "design [3]. This experiment uses a deliberately small simulator to make the "
            "trade-offs traceable.")
    section("3. Environment and evidence",
            "Six synthetic services receive pre-generated incident arrivals and telemetry. "
            "The defender can wait, inspect, isolate or restore. Isolation reduces exposed "
            "incident capacity, and dependency outages reduce service availability. A restore "
            "can clear an incident and incurs a temporary availability cost. No actions reach "
            "real services.")
    section("4. Experimental design",
            "Use four deterministic comparison policies and eight learned or imitation arms. "
            "Train DQN and PPO with three seeds. Select checkpoints on validation tapes. "
            "Evaluate 40 paired tapes each for independent, burst and held-out shifted regimes. "
            "Every final policy uses the same balanced objective. Reward variations affect "
            "training, and the evaluation retains each component.")
    graphic("tradeoffs.png",162)
    p("<b>Figure 1.</b> Mean availability and exposure for selected policies across three scenarios.")
    body.append(PageBreak())
    section("5. Held-out shifted results",
            "Higher return is better. Exposure and availability are value-weighted percentages. "
            "Invalid actions are mean counts within a 64-step episode.")
    rows=[["Policy","Exposure (%)","Availability (%)","Invalid","Return"]]
    for arm,name in POLICIES.items():
        rows.append([name,f"{metric(arm,'exposure')*100:.1f}",
                     f"{(1-metric(arm,'outage'))*100:.1f}",
                     f"{metric(arm,'invalid'):.1f}",reward(arm)])
    table=Table(rows,colWidths=[143,87,96,66,91],hAlign="LEFT")
    table.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"StudySans"),
         ("FONTNAME",(0,0),(-1,0),"StudySans-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
         ("BACKGROUND",(0,0),(-1,0),dark),("TEXTCOLOR",(0,0),(-1,0),colors.white),
         ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff5f7")]),
         ("ALIGN",(1,0),(-1,-1),"RIGHT"),("TOPPADDING",(0,0),(-1,-1),4.5),
         ("BOTTOMPADDING",(0,0),(-1,-1),4.5)]))
    body.append(table)
    graphic("learning.png",147)
    p("<b>Figure 2.</b> Checkpoint learning curves. Bands show the range across training seeds.")
    section("6. Failure analysis and limitations",
            f"The unmasked PPO chooses {metric('ppo','invalid'):.1f} invalid actions per "
            "shifted episode on average, compared with zero after masking. Compare rule-initialized "
            f"PPO (return {reward('ppo_masked_prior')}) against imitation only (return "
            f"{reward('imitation_only')}) to evaluate the additional learning step. "
            "The training budget, reward scale and synthetic alert distributions limit "
            "transfer to actual incident-response systems. The simulator models service states "
            "and incoming incidents without operational network actions.")
    p("References","Heading2")
    p("[1] Mnih et al., Nature (2015). [2] Schulman et al. (2017). "
      "[3] Standen et al. (2021). Full references appear in docs/METHODOLOGY.md.")
    p(f"Mean training time in this environment was {mean(x['seconds'] for x in data['training']):.2f}s "
      "across learned and imitation runs. The full evidence retains paired case results.")
    def footer(c,doc):
        c.setFont("StudySans",8)
        c.setFillColor(dark)
        c.drawString(52,26,"Reinforcement Learning for Defensive Incident Response")
        c.drawRightString(551,26,str(doc.page))
    document=SimpleDocTemplate(str(ROOT/"defensive-policy-report.pdf"),
                               title="Reinforcement Learning for Defensive Incident Response",
                               leftMargin=52,rightMargin=52,topMargin=42,bottomMargin=42)
    document.build(body,onFirstPage=footer,onLaterPages=footer)


if __name__=="__main__":
    with gzip.open(ROOT/"study.json.gz","rt",encoding="utf8") as file:
        data=json.load(file)
    index={(r["scenario"],r["model"]):r["metrics"] for r in data["summary"]}
    figures(data,index)
    pdf(data,index)
