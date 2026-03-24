from database import DatabaseManager


def main() -> int:
    db = DatabaseManager()
    return db.check_connection()


if __name__ == "__main__":
    raise SystemExit(main())
