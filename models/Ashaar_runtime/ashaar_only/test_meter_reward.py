# Quick smoke test for the Ashaar-only reward utilities.
from reward import AshaarRewardConfig, ashaar_reward


def main():
    # Example bayt with [sep] between shatrain
    text = "عرفتَ منازلَ أهلها فصَبت لهم واقتادها شجوٌ وطولُ شجونِ"

    cfg = AshaarRewardConfig(ashaar_weight=1.0)
    res = ashaar_reward(text, config=cfg)

    print("Text:", text)
    print("Ashaar-only reward:", res["reward"])
    print("Ashaar score:", res["ashaar_score"])


if __name__ == "__main__":
    main()
