from modules.hybrid_pos import HybridPOSTagger

def main():

    java_jar = "Library/FSPOST/stanford-postagger.jar"
    tagalog_model = "Library/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger"

    try:
        tagger = HybridPOSTagger(java_jar, tagalog_model)

        sentence = (
            "Dalawang kabataan sa Long Island, New York, na may edad na "
            "15 at 17, ay kinasuhan para sa kanilang di-umano'y "
            "pagkakasangkot sa isang hinihinalang plano na atakihin "
            "kanilang paaralan sa susunod na Abril, sa ika - siyam na "
            "anibersaryo ng pamamarilk sa marami sa Columbine High School."
        )

        tokens = tagger.tag(sentence)

        print(f"{len(tokens)} tokens\n")

        for token in tokens:
            print(
                f"{token.token:<20}"
                f"{str(token.penn_tag):<8}"
                f"{str(token.mgnn_tag):<10}"
            )

    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()