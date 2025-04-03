import {Pinecone} from "@pinecone-database/pinecone";
import {HfInference} from '@huggingface/inference'

const hf = new HfInference(process.env.HF_TOKEN)
export async function queryPineconeVectorStore(
  client: Pinecone,
  indexName: string,
  namespace: string,
  query: string
): Promise<string> {
  const apiOutput = await hf.featureExtraction({
    model: "mixedbread-ai/mxbai-embed-large-v1",
    inputs: query,
  });
  console.log(apiOutput);
  
  const queryEmbedding = Array.from(apiOutput);
  // console.log("Querying database vector store...");
  const index = client.Index(indexName);
  const queryResponse = await index.namespace(namespace).query({
    topK: 5,
    vector: queryEmbedding as any,
    includeMetadata: true,
    // includeValues: true,
    includeValues: false
  });

  console.log(queryResponse);


  if (queryResponse.matches.length > 0) {
    return queryResponse.matches.map((match, index) => {
      const metadata = match.metadata || {};
      return `
Molecular Research Passage ${index + 1}:
- Relevance Score: ${match.score?.toFixed(4) || 'N/A'}
- Source: ${metadata.source || 'Unknown'}
- Domain: ${metadata.domain || 'General'}

Passage Content:
${metadata.chunk || 'No detailed content'}

Key Insights:
- Molecular Target: ${metadata.molecularTarget || 'Not specified'}
- Research Focus: ${metadata.researchFocus || 'General'}
`;
    }).join("\n\n---\n\n");
  } else {
    return "No relevant molecular research passages found.";
  }
}


