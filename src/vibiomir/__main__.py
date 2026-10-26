from .cli import main

# Guard needed because the crawler's process pool re-imports __main__ on Windows (spawn).
if __name__ == "__main__":
    main()
