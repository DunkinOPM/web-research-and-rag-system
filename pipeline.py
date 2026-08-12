from agents import build_search_agent, writer_chain, critic_chain
from tools import scrape_url
from rag import (
    create_documents,
    split_documents,
    create_vector_store,
    create_retriever,
    retrieve_documents
)

import re


def extract_urls(search_results):
    urls = re.findall(
        r"https?://[^\s]+",
        search_results
    )

    cleaned_urls = []

    for url in urls:
        url = url.rstrip(".,;:)]}")

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    return cleaned_urls


def run_research_pipeline(topic: str) -> dict:

    state = {}

    print("\n" + "=" * 60)
    print("STEP 1 - SEARCH AGENT")
    print("=" * 60)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"""Research the following topic:

{topic}

Use the web search tool to find 5 recent, reliable and
relevant sources.

IMPORTANT:
Return the URLs exactly as provided by the web search tool.
Do not omit, modify, shorten or replace any URLs.

For every source provide:

Title:
URL:
Snippet:

Return all 5 sources."""
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("\nSEARCH RESULTS:\n")
    print(state["search_results"])


    print("\n" + "=" * 60)
    print("STEP 2 - EXTRACTING URLS")
    print("=" * 60)

    urls = extract_urls(
        state["search_results"]
    )

    urls = urls[:5]

    if not urls:
        raise RuntimeError(
            "The Search Agent did not return any usable URLs."
        )

    print(f"\nFound {len(urls)} URLs")

    for i, url in enumerate(urls):
        print(f"{i + 1}. {url}")


    print("\n" + "=" * 60)
    print("STEP 3 - SCRAPING SOURCES")
    print("=" * 60)

    scraped_documents = []

    for url in urls:

        print(f"\nScraping: {url}")

        content = scrape_url.invoke({
            "url": url
        })

        if content.startswith("Could not scrape"):
            print("Scraping failed")
            continue

        scraped_documents.append({
            "url": url,
            "content": content
        })

        print(
            f"Scraped {len(content)} characters"
        )

    state["scraped_documents"] = scraped_documents

    print(
        f"\nSuccessfully scraped "
        f"{len(scraped_documents)} sources"
    )

    if not scraped_documents:
        raise RuntimeError(
            "No sources could be scraped."
        )


    print("\n" + "=" * 60)
    print("STEP 4 - CREATING DOCUMENTS")
    print("=" * 60)

    state["documents"] = create_documents(
        state["scraped_documents"]
    )

    print(
        f"\nDocuments created: "
        f"{len(state['documents'])}"
    )


    print("\n" + "=" * 60)
    print("STEP 5 - CHUNKING DOCUMENTS")
    print("=" * 60)

    state["chunks"] = split_documents(
        state["documents"]
    )

    print(
        f"\nChunks created: "
        f"{len(state['chunks'])}"
    )

    if not state["chunks"]:
        raise RuntimeError(
            "No chunks were created."
        )


    print("\n" + "=" * 60)
    print("STEP 6 - CREATING VECTOR STORE")
    print("=" * 60)

    state["vector_store"] = create_vector_store(
        state["chunks"]
    )

    print("\nFAISS vector store created successfully")


    print("\n" + "=" * 60)
    print("STEP 7 - CREATING RETRIEVER")
    print("=" * 60)

    retriever = create_retriever(
        state["vector_store"]
    )

    print("\nRetriever created successfully")


    print("\n" + "=" * 60)
    print("STEP 8 - RETRIEVING RELEVANT RESEARCH")
    print("=" * 60)

    retrieved_docs = retrieve_documents(
        retriever,
        topic
    )

    state["retrieved_documents"] = retrieved_docs

    print(
        f"\nRetrieved "
        f"{len(retrieved_docs)} relevant chunks"
    )

    for i, doc in enumerate(retrieved_docs):

        print(f"\nRetrieved Chunk {i + 1}")

        print(
            f"Source: "
            f"{doc.metadata.get('source')}"
        )

        print(
            f"Content: "
            f"{doc.page_content[:300]}..."
        )


    print("\n" + "=" * 60)
    print("STEP 9 - WRITER")
    print("=" * 60)

    research_for_writer = ""

    for doc in retrieved_docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        research_for_writer += (
            f"\nSOURCE: {source}\n"
            f"{doc.page_content}\n"
            f"\n{'-' * 60}\n"
        )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_for_writer
    })

    print("\nFINAL REPORT:\n")
    print(state["report"])


    print("\n" + "=" * 60)
    print("STEP 10 - CRITIC")
    print("=" * 60)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCRITIC REPORT:\n")
    print(state["feedback"])

    return state


if __name__ == "__main__":

    topic = input(
        "\nEnter a research topic: "
    )

    run_research_pipeline(topic)