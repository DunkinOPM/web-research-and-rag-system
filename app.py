import streamlit as st
from pipeline import run_research_pipeline


st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide"
)


st.title("🔎 Multi-Agent Research System")

st.markdown(
    """
    Enter a research topic and let the system search the web,
    scrape relevant sources, retrieve useful information using RAG,
    generate a research report, and critique the result.
    """
)


st.divider()


topic = st.text_input(
    "Research Topic",
    placeholder="e.g. Impact of AI agents on software development"
)


research_button = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True
)


if research_button:

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        with st.status(
            "Research system is working...",
            expanded=True
        ) as status:

            try:

                st.write("🔎 Searching the web...")
                
                result = run_research_pipeline(topic)

                status.update(
                    label="Research completed successfully!",
                    state="complete",
                    expanded=False
                )

            except Exception as e:

                status.update(
                    label="Research failed",
                    state="error"
                )

                st.error(
                    f"An error occurred: {str(e)}"
                )

                st.stop()


        st.divider()

        st.header("📄 Research Report")

        st.markdown(
            result["report"]
        )


        st.divider()

        st.header("🔍 Sources")

        scraped_documents = result.get(
            "scraped_documents",
            []
        )

        if scraped_documents:

            for i, document in enumerate(
                scraped_documents,
                start=1
            ):

                url = document["url"]

                st.markdown(
                    f"**Source {i}:** [{url}]({url})"
                )

        else:

            st.info("No sources were available.")


        st.divider()

        st.header("🧠 Retrieved Research")

        retrieved_documents = result.get(
            "retrieved_documents",
            []
        )

        if retrieved_documents:

            for i, document in enumerate(
                retrieved_documents,
                start=1
            ):

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                with st.expander(
                    f"Retrieved Chunk {i}"
                ):

                    st.markdown(
                        f"**Source:** [{source}]({source})"
                    )

                    st.write(
                        document.page_content
                    )

        else:

            st.info(
                "No retrieved documents were returned."
            )


        st.divider()

        st.header("📝 Critic Evaluation")

        feedback = result.get(
            "feedback",
            ""
        )

        st.markdown(feedback)


        with st.expander("View Search Results"):

            st.write(
                result.get(
                    "search_results",
                    "No search results available."
                )
            )