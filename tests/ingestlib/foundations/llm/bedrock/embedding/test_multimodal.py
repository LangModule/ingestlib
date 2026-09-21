"""Cross-modal semantic ordering — the whole point of multimodal embeddings.

Probes describe the REAL fixtures: photo.jpg is a cat photograph,
document_text.png is a driver's license, document_chart.png is a monthly
sales chart. Measured margins are wide (right pairing ~0.3-0.4 cosine,
wrong ~0.08), so these orderings are stable, not lucky.
"""


def test_text_cat_closer_to_photo_than_to_chart(
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


def test_text_license_closer_to_license_image_than_to_photo(
    embed, cos_sim, doc_text_path, photo_path
):
    v_text = embed.text("a driver's license identity card with a photo, name, and address")
    v_doc = embed.image(doc_text_path, detail_level="DOCUMENT_IMAGE")
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")

    sim_correct = cos_sim(v_text, v_doc)
    sim_wrong = cos_sim(v_text, v_photo)

    assert sim_correct > sim_wrong, (
        f'"license" text↔license sim {sim_correct:.4f} not greater than '
        f'"license" text↔photo sim {sim_wrong:.4f}'
    )


def test_text_sales_chart_closer_to_chart_image_than_to_photo(
    embed, cos_sim, doc_chart_path, photo_path
):
    v_text = embed.text("a bar chart of monthly sales showing units sold per month")
    v_chart = embed.image(doc_chart_path, detail_level="DOCUMENT_IMAGE")
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")

    sim_correct = cos_sim(v_text, v_chart)
    sim_wrong = cos_sim(v_text, v_photo)

    assert sim_correct > sim_wrong, (
        f'"sales chart" text↔chart sim {sim_correct:.4f} not greater than '
        f'"sales chart" text↔photo sim {sim_wrong:.4f}'
    )


def test_wrong_text_does_not_beat_right_text_for_photo(
    embed, cos_sim, photo_path
):
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")
    v_right = embed.text("a photograph of a domestic cat sitting outdoors")
    v_wrong = embed.text("a bar chart of monthly sales showing units sold per month")

    sim_right = cos_sim(v_photo, v_right)
    sim_wrong = cos_sim(v_photo, v_wrong)

    assert sim_right > sim_wrong, (
        f"photo↔cat_text sim {sim_right:.4f} not greater than "
        f"photo↔chart_text sim {sim_wrong:.4f}"
    )
