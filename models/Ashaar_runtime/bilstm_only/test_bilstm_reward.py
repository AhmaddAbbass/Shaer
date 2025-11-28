# Quick smoke test for the BiLSTM-only reward utilities.
from reward import BilstmRewardConfig, bilstm_reward

def main():
    text = "شَهيدٌ بِنَفْسٍ راسِخٍ عَزْم لِرَبِّهِ # مَقامٌ خُلودٍ عِندْ إِلَهٍ لِخُلْدِهِ"

    # This should match the classifier's label string for the desired meter
    target_meter = "الطويل"

    cfg = BilstmRewardConfig(classifier_target_label=target_meter)

    res = bilstm_reward(text, config=cfg)

    print("Text:", text)
    print("Target meter:", target_meter)
    print("Reward:", res["reward"])
    print("BiLSTM classifier score:", res["classifier_score"])
    print("Classifier distribution:", res["classifier_distribution"])

if __name__ == "__main__":
    main()
