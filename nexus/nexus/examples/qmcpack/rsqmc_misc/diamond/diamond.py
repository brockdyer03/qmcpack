#! /usr/bin/env python3
from pathlib import Path

from nexus import settings, job, run_project, PseudoSet
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack
from nexus import generate_qmcpack,vmc

pseudo_dir = Path("../../pseudopotentials").resolve()

settings(
    pseudo_dir    = pseudo_dir,
    status_only   = 0,
    generate_only = 0,
    sleep         = 3,
    machine       = 'ws16'
    )

espresso_pseudos = PseudoSet.from_dir(pseudo_dir, code="espresso")
qmcpack_pseudos = PseudoSet.from_dir(pseudo_dir, code="qmcpack")

dia16 = generate_physical_system(
    units  = 'A',
    axes   = [[ 1.785,  1.785,  0.   ],
              [ 0.   ,  1.785,  1.785],
              [ 1.785,  0.   ,  1.785]],
    elem   = ['C','C'],
    pos    = [[ 0.    ,  0.    ,  0.    ],
              [ 0.8925,  0.8925,  0.8925]],
    tiling = (2,2,2),
    kgrid  = (1,1,1),
    kshift = (0,0,0),
    C      = 4
    )

scf = generate_pwscf(
    identifier   = 'scf',
    path         = 'diamond/scf',
    job          = job(cores=16,app='pw.x'),
    input_type   = 'generic',
    calculation  = 'scf',
    input_dft    = 'lda',
    ecutwfc      = 200,
    conv_thr     = 1e-8,
    nosym        = True,
    wf_collect   = True,
    system       = dia16,
    pseudos      = espresso_pseudos,
    )

conv = generate_pw2qmcpack(
    identifier   = 'conv',
    path         = 'diamond/scf',
    job          = job(cores=1,app='pw2qmcpack.x'),
    write_psir   = False,
    dependencies = (scf,'orbitals'),
    )

qmc = generate_qmcpack(
    driver       = 'legacy',
    identifier   = 'vmc',
    path         = 'diamond/vmc',
    job          = job(cores=16,threads=4,app='qmcpack'),
    input_type   = 'basic',
    system       = dia16,
    pseudos      = qmcpack_pseudos,
    jastrows     = [],
    calculations = [
        vmc(
            walkers     =   1,
            warmupsteps =  20,
            blocks      = 200,
            steps       =  10,
            substeps    =   2,
            timestep    =  .4
            )
        ],
    dependencies = (conv,'orbitals'),
    )

run_project()
