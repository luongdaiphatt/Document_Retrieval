document.addEventListener("DOMContentLoaded", function () {
  const searchButton = document.getElementById("search-button");
  const searchQuery = document.getElementById("search-query");
  const resultsDiv = document.getElementById("results");
  
  if (searchButton && searchQuery && resultsDiv) {
    searchButton.addEventListener("click", function () {
      const query = searchQuery.value;
      if (query) {
        fetch("/search", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query: query }),
        })
          .then((response) => response.json())
          .then((data) => {
            const resultsDiv = document.getElementById("results");
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
          })
          .catch((error) => {
            console.error("Error:", error);
          });
      }
    });
  }
});
