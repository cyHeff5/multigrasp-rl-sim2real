import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"deploy scaffold | config={args.config}")


if __name__ == "__main__":
    main()
