from src.bin import Converter
from src.utils import gather_settings


def main():
    settings = gather_settings()
    Converter(version=settings.version,
              size=settings.size).convert_to(framework=settings.convert_to)


if __name__ == '__main__':
    main()
