import dotenv from "dotenv";
dotenv.config();
import express from "express";
import cors from "cors";
import * as fs from "fs";
import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const app = express();
const port = process.env.PORT || 4000;
app.use(express.json());
app.use(cors());

/* Initialize the LLM to use to answer the question */
const model = new OpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });
/* Load in the file we want to do question answering over */
const text = fs.readFileSync("./Handbook1.txt", "utf8");
/* Split the text into chunks */
const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
const docs = await textSplitter.createDocuments([text]);
/* Create the vectorstore */
const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
/* Create the chain */
const chain = ConversationalRetrievalQAChain.fromLLM(
  model,
  vectorStore.asRetriever()
);

app.post("/chatbot", async (req, res) => {
  try {
    const { prompt } = req.body;
    const result = await chain.call({ question: prompt, chat_history: [] });

    const data = {
      question: prompt,
      response: result.text,
    };

    const filePath = "responses.txt";

    const textData = Object.entries(data)
      .map(([key, value]) => `\n${key}: ${value}`)
      .join("");

    fs.appendFile(filePath, textData, "utf8", (err) => {
      if (err) {
        console.error("An error occurred while writing the file:", err);
      } else {
        console.log("File has been successfully written!");
      }
    });

    res.json({
      message: result,
    });
  } catch (error) {
    res.json({ error });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
