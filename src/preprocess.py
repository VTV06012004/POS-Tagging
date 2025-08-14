def build_tag_mapping(all_tags):
    """
    Input:
        all_tags: List[List[str]] — list các câu, mỗi câu là list POS tags
    Output:
        tag2id: dict — POS tag → số
        id2tag: dict — số → POS tag
    """
    unique_tags = sorted({tag for tags in all_tags for tag in tags})
    
    tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    
    return tag2id, id2tag
