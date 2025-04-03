import { queryPineconeVectorStore } from "@/utils";
import { Pinecone } from "@pinecone-database/pinecone";
// import { Message, OpenAIStream, StreamData, StreamingTextResponse } from "ai";
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { generateText, Message, StreamData, streamText } from "ai";

// Allow streaming responses up to 30 seconds
export const maxDuration = 60;
// export const runtime = 'edge';

const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY ?? "",
});

const google = createGoogleGenerativeAI({
    baseURL: 'https://generativelanguage.googleapis.com/v1beta',
    apiKey: process.env.GEMINI_API_KEY
});

// gemini-1.5-pro-latest
// gemini-1.5-pro-exp-0801
const model = google('models/gemini-1.5-pro-latest', {
    safetySettings: [
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' }
    ],
});

export async function POST(req: Request, res: Response) {
    const reqBody = await req.json();
    console.log(reqBody);

    const messages: Message[] = reqBody.messages;
    const userQuestion = `${messages[messages.length - 1].content}`;

    const reportData: string = reqBody.data.reportData;
    const query = `  Molecular Research Context:
    - Disease: ${reportData}
    - Research Focus: ${reportData}
    
    Find relevant molecular interactions, potential drug candidates, and research insights.
    `;

    const retrievals = await queryPineconeVectorStore(pinecone, 'drug-discovery', "Default", query);

    const finalPrompt = `Drug Discovery Research Analysis

**Disease Context:** ${reportData}

**Research Objective:** ${userQuestion}

**Molecular Research Database Insights:**
${retrievals}

Research Guidance:
1. Provide a comprehensive analysis of potential drug candidates
2. Evaluate molecular interaction mechanisms
3. Assess therapeutic potential
4. Suggest structural modification strategies
5. Consider pharmacological challenges and opportunities

Analytical Framework:
- Leverage ChemBL database insights
- Apply Lipinski's Rule of Five
- Consider target protein interactions
- Evaluate binding affinity potential
- Analyze structural modifications

Deliver a structured, scientifically rigorous response that:
- Explains molecular interaction hypotheses
- Identifies promising drug candidate characteristics
- Provides evidence-based recommendations
- Highlights potential research directions

**Detailed Research Analysis:**
  `;

    const data = new StreamData();
    data.append({
        retrievals: retrievals
    });

    const result = await streamText({
        model: model,
        prompt: finalPrompt,
        onFinish() {
            data.close();
        }
    });

    return result.toDataStreamResponse({ data });
}

