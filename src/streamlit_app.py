import streamlit as st
from predict import predict_tags

st.set_page_config(page_title="POS Tagging Online", layout="wide")

# Custom CSS (to preserve your index.html style)
st.markdown("""
    <style>
    body {
      font-family: Arial, sans-serif;
      background: #f9f9f9;
    }
    .btn {
      margin-top: 10px;
      padding: 10px 20px;
      font-size: 16px;
      border: none;
      border-radius: 8px;
      background: #4CAF50;
      color: white;
      cursor: pointer;
    }
    .btn:hover {
      background: #45a049;
    }
    .tag {
      padding: 3px 6px;
      border-radius: 4px;
      margin: 2px;
      display: inline-block;
      cursor: pointer;
      color: #000;
    }
    .adj { background: #FFD966; }
    .adv { background: #E6B8AF; }
    .conj { background: #D9D2E9; }
    .det { background: #C9DAF8; }
    .noun { background: #B4A7D6; }
    .num { background: #93C47D; }
    .prep { background: #F6B26B; }
    .pron { background: #FFE599; }
    .verb { background: #B6D7A8; }
    .propn { background: #6FA8DC; }
    .other { background: #CCCCCC; }
    #legend span {
      display: inline-block;
      padding: 5px 10px;
      margin: 5px;
      border-radius: 4px;
      font-size: 14px;
      color: #000;
    }
    </style>
""", unsafe_allow_html=True)

st.title("POS Tagging Online")
st.write("Nhập câu của bạn và nhấn nút để phân tích từ loại:")

# Input
text = st.text_area("Nhập câu tại đây...", height=100)

# Button
if st.button("Phân tích POS", key="analyze"):
    if not text.strip():
        st.warning("⚠️ Vui lòng nhập câu!")
    else:
        tags = predict_tags(text.strip())

        # Result container
        st.markdown("### 👉 Kết quả:")
        result_html = ""
        posColors = {
            "ADJ": "adj",
            "ADV": "adv",
            "CCONJ": "conj",
            "SCONJ": "conj",
            "DET": "det",
            "NOUN": "noun",
            "NUM": "num",
            "ADP": "prep",
            "PRON": "pron",
            "VERB": "verb",
            "PROPN": "propn"
        }

        for word, pos in tags:
            css_class = posColors.get(pos, "other")
            result_html += f"<span class='tag {css_class}' title='{word} → {pos}'>{word}</span> "

        st.markdown(result_html, unsafe_allow_html=True)

# Legend
st.markdown("### Legend")
legend_html = """
<span class="adj">Adjective</span>
<span class="adv">Adverb</span>
<span class="conj">Conjunction</span>
<span class="det">Determiner</span>
<span class="noun">Noun</span>
<span class="num">Number</span>
<span class="prep">Preposition</span>
<span class="pron">Pronoun</span>
<span class="verb">Verb</span>
<span class="propn">Proper Noun</span>
"""
st.markdown(legend_html, unsafe_allow_html=True)
