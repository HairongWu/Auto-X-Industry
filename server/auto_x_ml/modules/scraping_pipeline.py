from scrapegraphai.graphs import SearchGraph

class ScrapingPipeline(): 
    def __init__(self):
        super().__init__()

    def scraping_web(self, prompt, source):
        # Define the configuration for the graph
        graph_config = {
             "llm": {
                "model": "ollama/mistral",
                "temperature": 0,
                "format": "json",  # Ollama needs the format to be specified explicitly
                "base_url": "http://localhost:11434",  # set Ollama URL
            },
            "embeddings": {
                "model": "ollama/nomic-embed-text",
                "base_url": "http://localhost:11434",  # set Ollama URL
            },
            "verbose": True,
        }

        # Create the SearchGraph instance
        self.search_graph = SearchGraph(
            prompt=prompt,
            source=source,
            config=graph_config
        )
        # Run the graph
        result = self.search_graph.run()
        return result
    
    def search_engine(self):
        pass