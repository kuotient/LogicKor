# LogicKor
한국어 언어모델 다분야 사고력 벤치마크

## Benchmark Website
https://lk.instruct.kr/

## Current Benchmarks
| 이름 | # | 데이터 | 출처 |
|---|---|---|---|
| `logic-kor` | 42 | 한국어 언어모델 다분야 사고력 벤치마크 | instruct-kr |
| `mt-bench-ko` | 80 | [MT-BENCH](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)의 한국어 번역, 검수 벤치마크 데이터셋 | [kuotient/FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |

## Repository
본 Repo는 LogicKor를 포함한 다양한 다분야 사고력 벤치마크의 추론 및 평가 코드, 데이터셋을 담고 있습니다.

## Evalutation Example
EEVE 템플릿, GPU 0,1 사용, model_len 4096, logic-kor 벤치마크
```
python generator.py --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --template templates/template-EEVE.json --bench_name logic-kor --gpu_devices 0,1 --model_len 4096
python judgement.py --model_output logic-kor/yanolja_EEVE-Korean-Instruct-10.8B-v1.0.jsonl --openai_api_key sk-somethingsomething --threads 30
```
