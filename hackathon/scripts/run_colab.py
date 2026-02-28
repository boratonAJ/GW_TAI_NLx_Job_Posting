import subprocess
import sys


def main() -> None:
    try:
        from pyngrok import ngrok
    except ImportError as exc:
        raise SystemExit(
            "Install pyngrok first: pip install pyngrok"
        ) from exc

    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "hackathon/app.py"])
    ngrok.kill()
    public_url = ngrok.connect(8501)
    print(f"LIVE APP: {public_url}")


if __name__ == "__main__":
    main()
