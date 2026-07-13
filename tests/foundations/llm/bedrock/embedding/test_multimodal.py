"""Cross-modal semantic ordering — the whole point of multimodal embeddings."""


def test_text_cat_closer_to_photo_than_to_periodic_table(
    embed, cos_sim, photo_path, doc_chart_path
):
    v_text = embed.text("a photograph of a domestic cat sitting outdoors")
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")
    v_chart = embed.image(doc_chart_path, detail_level="DOCUMENT_IMAGE")

    sim_correct = cos_sim(v_text, v_photo)
    sim_wrong = cos_sim(v_text, v_chart)

    assert sim_correct > sim_wrong, (
        f'"cat" text↔photo sim {sim_correct:.4f} not greater than '
        f'"cat" text↔chart sim {sim_wrong:.4f}'
    )


def test_text_form_closer_to_document_page_than_to_photo(
    embed, cos_sim, doc_text_path, photo_path
):
    v_text = embed.text("an employment application form with fields for name, address, and job history")
    v_doc = embed.image(doc_text_path, detail_level="DOCUMENT_IMAGE")
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")

    sim_correct = cos_sim(v_text, v_doc)
    sim_wrong = cos_sim(v_text, v_photo)

    assert sim_correct > sim_wrong, (
        f'"form" text↔doc_text sim {sim_correct:.4f} not greater than '
        f'"form" text↔photo sim {sim_wrong:.4f}'
    )


def test_text_periodic_table_closer_to_chart_image_than_to_photo(
    embed, cos_sim, doc_chart_path, photo_path
):
    v_text = embed.text("the periodic table of chemical elements with rows and columns")
    v_chart = embed.image(doc_chart_path, detail_level="DOCUMENT_IMAGE")
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")

    sim_correct = cos_sim(v_text, v_chart)
    sim_wrong = cos_sim(v_text, v_photo)

    assert sim_correct > sim_wrong, (
        f'"periodic table" text↔chart sim {sim_correct:.4f} not greater than '
        f'"periodic table" text↔photo sim {sim_wrong:.4f}'
    )


def test_wrong_text_does_not_beat_right_text_for_photo(
    embed, cos_sim, photo_path
):
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")
    v_right = embed.text("a photograph of a domestic cat sitting outdoors")
    v_wrong = embed.text("the periodic table of chemical elements with rows and columns")

    sim_right = cos_sim(v_photo, v_right)
    sim_wrong = cos_sim(v_photo, v_wrong)

    assert sim_right > sim_wrong, (
        f"photo↔cat_text sim {sim_right:.4f} not greater than "
        f"photo↔periodic_table_text sim {sim_wrong:.4f}"
    )
