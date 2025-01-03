document.addEventListener('DOMContentLoaded', function() {
  let currentPage = 1;
  const maxPage = 30;

  const searchButton = document.getElementById("search-button");
  const searchQuery = document.getElementById("search-query");
  const resultsDiv = document.getElementById("results");
  const paginationContainer = document.querySelector('.pagination');

  const previousButton = document.createElement('button');
  previousButton.className = 'previous-button';
  previousButton.textContent = 'Previous';
  previousButton.addEventListener('click', () => {
    if (currentPage > 1) {
      currentPage--;
      fetchResults();
    }
  });

  const nextButton = document.createElement('button');
  nextButton.className = 'next-button';
  nextButton.textContent = 'Next';
  nextButton.addEventListener('click', () => {
    if (currentPage < maxPage) {
      currentPage++;
      fetchResults();
    }
  });

  function renderPagination() {
    // Remove existing page buttons
    const existingPageButtons = document.querySelectorAll('.page-button');
    existingPageButtons.forEach(button => button.remove());

    // Determine the range of pages to display
    let startPage = currentPage;
    let endPage = Math.min(maxPage, currentPage + 2);

    for (let i = startPage; i <= endPage; i++) {
      const pageButton = document.createElement('button');
      pageButton.className = 'page-button';
      pageButton.textContent = i;
      if (i === currentPage) {
        pageButton.classList.add('active');
      }
      pageButton.addEventListener('click', () => {
        currentPage = i;
        fetchResults();
      });
      paginationContainer.insertBefore(pageButton, nextButton);
    }
  }

  function fetchResults() {
    const query = searchQuery.value;
    if (query) {
      fetch("/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query, page: currentPage }),
      })
        .then((response) => response.json())
        .then((data) => {
          resultsDiv.innerHTML = ""; // Clear previous results

          data.results.forEach((result) => {
            // Create article result structure
            const article = document.createElement("article");
            article.classList.add("article-result");
            
            // Article Left (image)
            const articleLeft = document.createElement("div");
            articleLeft.classList.add("article-left");
            const img_a = document.createElement("a");
            img_a.href = result.link;
            const img = document.createElement("img");
            img.classList.add("article-image");
            img.src = result.imglink || "https://via.placeholder.com/100"; // Default image if not provided
            img.alt = "Article Image";
            img_a.appendChild(img);
            articleLeft.appendChild(img_a);

            // Article Right (details)
            const articleRight = document.createElement("div");
            articleRight.classList.add("article-right");

            const title = document.createElement("h3");
            title.classList.add("article-title");
            title.textContent = result.title;
            const title_a = document.createElement("a");
            title_a.classList.add("article-title-a");
            title_a.href = result.link;
            title_a.appendChild(title);

            const metadata = document.createElement("div");
            metadata.classList.add("article-metadata");

            const source = document.createElement("p");
            source.classList.add("article-source");
            source.textContent = result.source || "Unknown source";
            
            const topic = document.createElement("p");
            topic.classList.add("article-topic");
            topic.textContent = result.topic || "Unknown topic";

            metadata.appendChild(source);
            metadata.appendChild(topic);

            const abstract = document.createElement("p");
            abstract.classList.add("article-abstract");
            abstract.textContent = result.abstract;
            const abstract_a = document.createElement("a");
            abstract_a.href = result.link;
            abstract_a.classList.add("article-abstract-a");
            abstract_a.appendChild(abstract);

            // Append title, metadata, and abstract to articleRight
            articleRight.appendChild(title_a);
            articleRight.appendChild(metadata);
            articleRight.appendChild(abstract_a);

            // Append left and right sections to article
            article.appendChild(articleLeft);
            article.appendChild(articleRight);

            const datetime = document.createElement("p");
            datetime.classList.add("article-datetime");
            datetime.textContent = result.time || "Unknown datetime";

            articleRight.appendChild(datetime);

            // Add the complete article to the results div
            resultsDiv.appendChild(article);
          });

          // Render pagination buttons after fetching results
          renderPagination();
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    }
  }

  if (searchButton && searchQuery && resultsDiv) {
    searchButton.addEventListener("click", function () {
      currentPage = 1; // Reset to first page on new search
      fetchResults();
    });
  }

  // Append pagination buttons to the container
  paginationContainer.appendChild(previousButton);
  paginationContainer.appendChild(nextButton);
});


