#!/usr/bin/env ts-node
/**
 * Prompt Translator System
 *
 * 知識レベルの低い人の質問を理解し、プロレベルのプロンプトに変換し、
 * その回答を入力者のレベルに合わせた分かりやすい説明で返す3ステップシステム。
 *
 * Step 1: ユーザー入力 → 知識レベル分析 + プロ向けプロンプトへ変換
 * Step 2: プロ向けプロンプト → 専門家レベルの回答を取得
 * Step 3: 専門家回答 → ユーザーのレベルに合わせた分かりやすい説明に変換
 */

import Anthropic from "@anthropic-ai/sdk";
import * as readline from "readline";

const client = new Anthropic();

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface TranslationResult {
  level: number;          // 1〜5（1=超初心者、5=上級者）
  intent: string;         // ユーザーが本当に聞きたいこと（日本語）
  professionalPrompt: string;  // プロ向けに変換されたプロンプト
  originalLanguage: string;    // 入力言語
}

// ─────────────────────────────────────────────
// Step 1: ユーザー入力を分析し、プロプロンプトに変換
// ─────────────────────────────────────────────

async function translateToProPrompt(userInput: string): Promise<TranslationResult> {
  const response = await client.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 2048,
    thinking: { type: "adaptive" },
    system: `You are an expert prompt engineer and communication specialist.

Your task is to:
1. Analyze the knowledge level of the user's input (scale 1-5, where 1=complete beginner, 5=expert)
2. Deeply understand what the user truly wants to know, even if expressed poorly
3. Transform their simple/unclear question into a highly professional, technically precise, comprehensive prompt that will elicit the best possible expert answer from an AI

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{
  "level": <number 1-5>,
  "intent": "<what the user truly wants to know, in their original language>",
  "professionalPrompt": "<the transformed professional-grade prompt in English, technically precise, with proper context, constraints, and expected output format>",
  "originalLanguage": "<detected language of the input, e.g. 'Japanese', 'English'>"
}

Guidelines for the professional prompt:
- Use precise technical terminology
- Specify the exact type of response needed (e.g., step-by-step, comparative analysis, with examples)
- Add relevant context and constraints
- Request concrete, actionable information
- Target a domain expert as the audience
- Make it 3-5x more detailed and precise than the original`,

    messages: [
      {
        role: "user",
        content: `Analyze and transform this user input:\n\n"${userInput}"`,
      },
    ],
  });

  // Extract text from response
  let jsonText = "";
  for (const block of response.content) {
    if (block.type === "text") {
      jsonText = block.text;
      break;
    }
  }

  // Strip markdown code fences if present
  jsonText = jsonText.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();

  try {
    return JSON.parse(jsonText) as TranslationResult;
  } catch {
    throw new Error(`Failed to parse translation result: ${jsonText}`);
  }
}

// ─────────────────────────────────────────────
// Step 2: プロプロンプトで専門家回答を取得（ストリーミング）
// ─────────────────────────────────────────────

async function getExpertAnswer(professionalPrompt: string): Promise<string> {
  let expertAnswer = "";

  const stream = client.messages.stream({
    model: "claude-opus-4-6",
    max_tokens: 8192,
    thinking: { type: "adaptive" },
    system: `You are a world-class expert consultant. Provide comprehensive, technically accurate,
and deeply insightful answers. Include concrete examples, best practices, potential pitfalls,
and actionable recommendations. Structure your response clearly with relevant sections.`,
    messages: [
      { role: "user", content: professionalPrompt },
    ],
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      expertAnswer += event.delta.text;
    }
  }

  return expertAnswer;
}

// ─────────────────────────────────────────────
// Step 3: 専門家回答をユーザーレベルに合わせて説明（ストリーミング）
// ─────────────────────────────────────────────

async function simplifyForUser(
  expertAnswer: string,
  translation: TranslationResult,
  onChunk: (chunk: string) => void
): Promise<void> {
  const levelDescriptions: Record<number, string> = {
    1: "a complete beginner with no technical background",
    2: "someone with basic knowledge but limited experience",
    3: "someone with moderate understanding of the topic",
    4: "someone fairly knowledgeable but not yet an expert",
    5: "a knowledgeable person who just needed expert confirmation",
  };

  const levelDesc = levelDescriptions[translation.level] ?? levelDescriptions[3];

  const stream = client.messages.stream({
    model: "claude-opus-4-6",
    max_tokens: 4096,
    system: `You are a gifted teacher who excels at explaining complex topics simply.

The person you're explaining to is: ${levelDesc} (knowledge level ${translation.level}/5).
Their original question was about: ${translation.intent}
Respond in: ${translation.originalLanguage}

Rules:
- Use simple, everyday language appropriate for their level
- Avoid jargon; if technical terms are necessary, explain them immediately in parentheses
- Use analogies and real-world examples they can relate to
- Structure the answer with clear headings and short paragraphs
- Be encouraging and positive
- Focus on the most important and practical points
- Length should match complexity: don't overwhelm a beginner with every detail
- End with a simple summary of the key takeaway`,

    messages: [
      {
        role: "user",
        content: `Here is the expert answer to translate into simple terms:\n\n${expertAnswer}\n\nNow explain this in ${translation.originalLanguage} for someone at knowledge level ${translation.level}/5.`,
      },
    ],
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      onChunk(event.delta.text);
    }
  }
}

// ─────────────────────────────────────────────
// Display helpers
// ─────────────────────────────────────────────

function printSeparator(char = "─", length = 60): void {
  console.log(char.repeat(length));
}

function printHeader(text: string): void {
  printSeparator();
  console.log(`  ${text}`);
  printSeparator();
}

function getLevelBar(level: number): string {
  const filled = "█".repeat(level);
  const empty = "░".repeat(5 - level);
  return `${filled}${empty} (${level}/5)`;
}

// ─────────────────────────────────────────────
// Main pipeline
// ─────────────────────────────────────────────

async function runTranslator(userInput: string): Promise<void> {
  console.log("\n");
  printHeader("🔍 STEP 1 / ステップ1: 質問を分析中...");
  console.log("");

  // Step 1
  let translation: TranslationResult;
  try {
    translation = await translateToProPrompt(userInput);
  } catch (err) {
    console.error("❌ 分析中にエラーが発生しました:", err);
    throw err;
  }

  console.log(`📊 知識レベル: ${getLevelBar(translation.level)}`);
  console.log(`🎯 あなたの質問の意図: ${translation.intent}`);
  console.log("");
  console.log("✨ プロ向けプロンプトに変換しました:");
  console.log(`   "${translation.professionalPrompt.slice(0, 120)}..."`);

  console.log("\n");
  printHeader("🧠 STEP 2 / ステップ2: 専門家レベルの回答を生成中...");
  console.log("");

  // Step 2
  let expertAnswer: string;
  try {
    expertAnswer = await getExpertAnswer(translation.professionalPrompt);
  } catch (err) {
    console.error("❌ 専門家回答の生成中にエラーが発生しました:", err);
    throw err;
  }

  console.log(`✅ 専門家回答を取得しました (${expertAnswer.length.toLocaleString()} 文字)`);

  console.log("\n");
  printHeader("💬 STEP 3 / ステップ3: あなたのレベルに合わせて説明します");
  console.log("");
  printSeparator("═");
  console.log("");

  // Step 3 (streaming to console in real-time)
  try {
    await simplifyForUser(expertAnswer, translation, (chunk) => {
      process.stdout.write(chunk);
    });
  } catch (err) {
    console.error("\n❌ 説明の生成中にエラーが発生しました:", err);
    throw err;
  }

  console.log("\n");
  printSeparator("═");
  console.log("");
}

// ─────────────────────────────────────────────
// Interactive REPL mode
// ─────────────────────────────────────────────

async function runInteractive(): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log("");
  printSeparator("═");
  console.log("  🌟 プロンプト変換システム / Prompt Translator System");
  printSeparator("═");
  console.log("");
  console.log("  どんな質問でも、プロのレベルに変換して答えます。");
  console.log("  気軽に、思ったままの言葉で質問してください！");
  console.log("");
  console.log('  終了するには "exit" または "quit" と入力してください。');
  console.log("");

  const askQuestion = (): void => {
    rl.question("あなたの質問 > ", async (input) => {
      const trimmed = input.trim();

      if (!trimmed) {
        askQuestion();
        return;
      }

      if (["exit", "quit", "q", "終了"].includes(trimmed.toLowerCase())) {
        console.log("\nご利用ありがとうございました！\n");
        rl.close();
        return;
      }

      try {
        await runTranslator(trimmed);
      } catch {
        console.error("\n予期しないエラーが発生しました。もう一度お試しください。\n");
      }

      console.log("");
      askQuestion();
    });
  };

  askQuestion();
}

// ─────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("❌ エラー: ANTHROPIC_API_KEY 環境変数が設定されていません。");
    console.error("   export ANTHROPIC_API_KEY=your-api-key");
    process.exit(1);
  }

  if (args.length === 0) {
    // Interactive mode
    await runInteractive();
  } else {
    // Single-shot mode: question passed as CLI argument
    const userInput = args.join(" ");
    await runTranslator(userInput);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
