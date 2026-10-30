import os
import csv


def build_corpus():

    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    dataset_dir = os.path.join(base_dir, "dataset")

    kaggle_file = os.path.join(
        dataset_dir,
        "unlabeledTrainData.tsv"
    )

    project_file = os.path.join(
        dataset_dir,
        "project_domain.txt"
    )

    output_file = os.path.join(
        dataset_dir,
        "corpus.txt"
    )

    print("Building final corpus...")
    print()

    total_reviews = 0

    with open(output_file, "w", encoding="utf-8") as output:

        # Kaggle IMDB data
        print("Reading Kaggle IMDB dataset...")

        with open(
            kaggle_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            reader = csv.DictReader(file, delimiter="\t")

            for row in reader:

                review = row.get("review", "").strip()

                if review:
                    output.write(review)
                    output.write("\n")
                    total_reviews += 1

        print("Kaggle reviews added:", total_reviews)

        # Our project-specific corpus
        print()
        print("Adding project domain corpus...")

        with open(
            project_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            project_text = file.read()

        output.write(project_text)

    print()
    print("Final corpus created successfully!")
    print("Kaggle reviews:", total_reviews)
    print("Project domain corpus: added")
    print()
    print("Corpus saved at:")
    print(output_file)


if __name__ == "__main__":
    build_corpus()