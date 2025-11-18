import os


def main():
    # delete savegame.json
    savegame_path = os.path.join(os.getcwd(), "savegame.json")
    if os.path.exists(savegame_path):
        os.remove(savegame_path)
        print(f"Deleted {savegame_path}")
    else:
        print(f"No savegamee found at {savegame_path}")
    # duplicate and rename backup.json to sacegame.json
    backup_path = os.path.join(os.getcwd(), "backup.json")
    if os.path.exists(backup_path):
        # duplicate
        with open(backup_path, "r") as backup_file:
            data = backup_file.read()
        with open(savegame_path, "w") as savegame_file:
            savegame_file.write(data)
        print(f"Restored backup from {backup_path} to {savegame_path}")


if __name__ == "__main__":
    main()
