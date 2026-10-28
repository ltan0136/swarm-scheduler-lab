# Third-party sources and notices

This project implements its own restricted rule search, evaluator, validation,
budget accounting and experiment orchestration. It calls public library APIs;
it does not vendor the libraries or copy their example implementations.
Dependency licenses remain applicable to the installed packages.

## JobShopLib 1.7.2

- Project: https://github.com/Pabloo22/job_shop_lib
- License: MIT
- Copyright: 2024 Pablo Ariño
- Dispatching example: https://job-shop-lib.readthedocs.io/en/stable/examples/03-Dispatching-Rules.html

FT06 is loaded from JobShopLib's packaged public benchmark dataset. FT06
originates from the Fisher and Thompson job-shop benchmarks (1963). Its known
optimal makespan, 55, is used only as a public benchmark reference, never as
Agent feedback or an optimization target in prompts. The package is the source
of the instance data; generated demonstration results may serialize this
instance for reproducibility.

JobShopLib's license notice is reproduced here to accompany the benchmark
data included in reproducibility artifacts:

```text
MIT License

Copyright (c) 2024 Pablo Ariño

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OR-Tools 9.12.4544

- Project: https://github.com/google/or-tools
- License: Apache License 2.0
- License source: https://github.com/google/or-tools/blob/stable/LICENSE
- Official explanation: https://developers.google.com/optimization/scheduling/job_shop
- Official example used to verify interval/no-overlap API:
  https://github.com/google/or-tools/blob/stable/ortools/sat/samples/minimal_jobshop_sat.py

The CP-SAT reference solver is an independent implementation using the public
Python API. The OR-Tools package retains its own license and notices; its code
and example source are not copied into this repository.

## Other runtime dependencies

Matplotlib (Matplotlib license), jsonschema (MIT), and their dependencies are
installed as separately distributed packages. The exact installed versions
are recorded in `requirements-lock.txt`; each package retains its own license.
