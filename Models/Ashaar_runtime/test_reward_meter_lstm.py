# test_meter_score.py

from meter_reward_LSTM import meter_reward, MeterRewardConfig

def main():
    text = "عرفتَ منازلَ أهلها فصَبت لهم واقتادها شجوٌ وطولُ شجونِ"

    # This should match the classifier's label string for the desired meter
    target_meter = "الكامل"

    cfg = MeterRewardConfig(
        classifier_weight=0.7,
        ashaar_weight=0.3,
        classifier_target_label=target_meter,
    )

    res = meter_reward(text, config=cfg)

    print("Text:", text)
    print("Target meter:", target_meter)
    print("Combined reward:", res["reward"])
    print("BiLSTM classifier score:", res["classifier_score"])
    print("Ashaar score:", res["ashaar_score"])
    print("Classifier distribution:", res["classifier_distribution"])

if __name__ == "__main__":
    main()
