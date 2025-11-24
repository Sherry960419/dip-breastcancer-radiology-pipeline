from Bio import Entrez
from config import PUBMED_EMAIL


def run_step(context: dict) -> dict:
    """
    Search PubMed using "breast cancer" + (optional) tumor shape keyword.
    We request up to 20 most relevant papers and extract:
      - title
      - journal
      - year (if available)
    The result is stored in context["papers"] as a list of dicts.
    """
    shape = context.get("shape", "")
    # Base query: always include "breast cancer"
    if shape:
        query = f"breast cancer {shape}"
    else:
        query = "breast cancer"

    print(f"[Step4] Searching PubMed for: {query}")

    # Required by NCBI Entrez
    Entrez.email = PUBMED_EMAIL

    # esearch default sort is "relevance", but set it explicitly for clarity
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=20,        # ask up to 20 ids
        sort="relevance", # most relevant
    )
    result = Entrez.read(handle)
    handle.close()

    ids = result.get("IdList", [])
    print(f"[Step4] Retrieved {len(ids)} papers.")

    papers = []
    if not ids:
        context["papers"] = papers
        return {"papers": papers}

    # Fetch article details (XML)
    handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    for article in records.get("PubmedArticle", []):
        article_data = article.get("MedlineCitation", {}).get("Article", {})
        title = article_data.get("ArticleTitle", "")

        journal_info = article_data.get("Journal", {})
        journal_title = journal_info.get("Title", "")

        pub_date = journal_info.get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")
        # Sometimes only MedlineDate exists, e.g. "2024 Jan-Feb"
        if not year and "MedlineDate" in pub_date:
            year = str(pub_date["MedlineDate"])

        papers.append(
            {
                "title": str(title),
                "journal": str(journal_title),
                "year": str(year),
            }
        )

    context["papers"] = papers
    return {"papers": papers}
