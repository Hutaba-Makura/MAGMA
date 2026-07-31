#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.trg_memory import TemporalResonanceGraphMemory
from memory.graph_db import NetworkXGraphDB
from memory.vector_db import NumpyVectorDB
from memory.memory_builder import MemoryBuilder
from memory.query_engine import QueryEngine
from memory.answer_formatter import AnswerFormatter
from memory.test_harness import TestHarness
from memory.evaluator import Evaluator
from utils.memory_layer import LLMController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TRGSystem:
    def __init__(self, model="gpt-4o-mini", embedding_model="minilm", cache_dir="./cache"):
        self.model = model
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._init_components()

    def _init_components(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found - using fallback mode")
            self.llm_controller = None
        else:
            self.llm_controller = LLMController(
                backend='openai',
                model=self.model,
                api_key=api_key
            )

        self.graph_db = NetworkXGraphDB()
        self.vector_db = NumpyVectorDB()

        self.trg_memory = TemporalResonanceGraphMemory(
            graph_db=self.graph_db,
            vector_db=self.vector_db,
            embedding_model=self.embedding_model
        )

        self.memory_builder = MemoryBuilder(
            trg_memory=self.trg_memory,
            model=self.model,
            use_episodes=False
        )

        self.query_engine = QueryEngine(
            trg=self.trg_memory,
            model=self.model
        )

        self.answer_formatter = AnswerFormatter()

        self.test_harness = TestHarness(
            trg=self.trg_memory,
            query_engine=self.query_engine,
            model=self.model
        )

        self.evaluator = Evaluator(model=self.model)

    def build_memory_from_conversation(self, conversation_data):
        logger.info(f"Building memory from {len(conversation_data)} turns")
        return self.memory_builder.build_memory(conversation_data)

    def query(self, question):
        context_nodes = self.query_engine.query(question, top_k=5)
        context = self.answer_formatter.format_context(context_nodes)

        if self.llm_controller:
            prompt = self.answer_formatter.build_qa_prompt(context, question)
            response = self.llm_controller.get_completion(prompt)
            answer = self.answer_formatter.extract_answer(response, question)
        else:
            answer = self._extract_simple_answer(context, question)

        return answer

    def _extract_simple_answer(self, context, question):
        q_lower = question.lower()
        context_lower = context.lower()

        sentences = context.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in q_lower.split()):
                return sentence.strip()

        return "Unable to find answer in context"

    def save_memory(self, save_path=None):
        if save_path is None:
            save_path = self.cache_dir

        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)

        graph_path = save_path / "graph.json"
        self.trg_memory.save_to_file(str(graph_path))

        vectors_path = save_path / "vectors"
        vectors_path.mkdir(exist_ok=True)
        self.vector_db.save(str(vectors_path))

        logger.info(f"Memory saved to {save_path}")

    def load_memory(self, load_path=None):
        if load_path is None:
            load_path = self.cache_dir

        load_path = Path(load_path)

        graph_path = load_path / "graph.json"
        if graph_path.exists():
            self.trg_memory.load_from_file(str(graph_path))

        vectors_path = load_path / "vectors"
        if vectors_path.exists():
            self.vector_db.load(str(vectors_path))

        logger.info(f"Memory loaded from {load_path}")

def main():
    parser = argparse.ArgumentParser(description='TRG Memory System')
    parser.add_argument('--mode', choices=['build', 'query', 'test'],
                       default='test', help='Operation mode')
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--question', help='Question to ask')
    parser.add_argument('--model', default='gpt-4o-mini', help='LLM model')
    parser.add_argument('--embedding-model', default='minilm',
                       choices=['minilm', 'openai'], help='Embedding model')
    parser.add_argument('--cache-dir', default='./cache', help='Cache directory')

    args = parser.parse_args()

    load_dotenv()

    system = TRGSystem(
        model=args.model,
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir
    )

    if args.mode == 'build':
        if not args.input:
            print("Error: --input required for build mode")
            return

        with open(args.input, 'r') as f:
            data = json.load(f)

        system.build_memory_from_conversation(data)
        system.save_memory()

        print("Memory built and saved successfully")

    elif args.mode == 'query':
        if not args.question:
            print("Error: --question required for query mode")
            return

        system.load_memory()
        answer = system.query(args.question)

        print(f"Question: {args.question}")
        print(f"Answer: {answer}")

    elif args.mode == 'test':
        if not args.input:
            print("Error: --input required for test mode")
            return

        with open(args.input, 'r') as f:
            test_data = json.load(f)

        results = system.test_harness.run_tests(test_data)
        metrics = system.evaluator.evaluate_results(results)

        print(f"Test Results:")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"F1 Score: {metrics['f1']:.2%}")
        print(f"BLEU Score: {metrics['bleu']:.2%}")

if __name__ == "__main__":
    main()