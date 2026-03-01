from hackathon.core.data import prepare_nlp_artifacts, prepare_raw_data


if __name__ == "__main__":
    jobs_file, processed_file = prepare_raw_data()
    mentions_file, profiles_file, requirements_file = prepare_nlp_artifacts()
    print(f"Prepared: {jobs_file}")
    print(f"Prepared: {processed_file}")
    print(f"Built NLP mentions: {mentions_file}")
    print(f"Built NLP skill profiles: {profiles_file}")
    print(f"Built NLP requirements profile: {requirements_file}")
