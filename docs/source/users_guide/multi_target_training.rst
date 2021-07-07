Multi-Target Training
=====================

In the last tutorial we trained and deployed a model for a single target.  While
this in itself is of course interesting, the true power of Deepest Histology
only comes to light if we want to train a large amount of models at the same
time.  In this tutorial we will take a look at how to automatically train models
for a range of different targets and harness the power of multiple GPUs in the
progress.


Defining a Multi-Target Run Getter
----------------------------------

Let's start off by defining a simple run getter::

    simple_train_get = partial(
        get.simple_run,
        train_cohorts=train_cohorts,
        max_tile_num=100,
        na_values=['inconclusive'])

Unlike the run getter in the previous tutorial, we did not specify the
``target_label`` this time around.  This is because we actually don't want to
manually specify the run's target label, but automatically repeat the training
with different target labels.  To achieve this, we use a *run adapter*:  instead
of generating runs by itself, a run adapter takes another run getter and
transforms it;  in our example, we take a single target run getter and adapt it
into a multi-target one.  It is constructed as such::

    multi_train_get = partial(
        get.multi_target,
        simple_train_get,
        target_labels=['isMSIH', 'gender', ...] #TODO add some more target labels
    )

The rest of the file looks similar last time::

    if __name__ == '__main__':
        do_experiment(
            project_dir='/path/to/training/project/dir',
            get=multi_train_get,
            devices=['cuda:0', 'cuda:1'],
            num_concurrent_runs=2,
            num_workers=4)  # set to 0 on Windows!

There are two new additions this time:  First of all, we specify to train up to
two models in parallel, using both GPUs in our system.  Each of these runs will
have four of the CPU's cores assigned to it for data preprocessing.  This way,
we can easily parallelize the training of multiple models and thus decrease the
overall training time.  On Windows, the ``num_workers`` option is currently
broken due to a bug in the pytorch module internally used by Deepest Histology.
On such machines it may be helpful set ``num_workers`` to zero and further
increase the number of concurrent runs.

When running this script, we can the directory structure is as follows::

    /path/to/training/project/dir
    ├── isMSIH
    │   └── export.pkl
    └── gender
         └── export.pkl

When looking at the previous tutorial, the model (the ``export.pkl`` file) was
saved directly in the project directory.  The ``multi_target`` adapter added an
additional subdirectory with the target's name to the project directory for each
target.


Evaluating Multiple Targets
---------------------------

The deployment script is modified in almost the same way as the training
script::

    simple_deploy_get = partial(
        get.simple_run,
        test_cohorts=test_cohorts,
        max_tile_num=100,
        na_values=['inconclusive'])

    multi_deploy_get = partial(
        get.multi_target,
        simple_deploy_get,
        target_labels=['isMSIH', 'gender'])

    project_dir='/path/to/deployment/project/dir',

    if __name__ == '__main__':
        do_experiment(
            project_dir=project_dir,
            get=multi_deploy_get,
            train=partial(
                get.load,
                project_dir=project_dir,
                training_project_dir='/path/to/training/project/dir')
            evaluator_groups=[
                [],
                [Grouped(auroc), Grouped(count)]
            ])

Evaluator Groups
^^^^^^^^^^^^^^^^

The attentive reader will have seen that the structure of the
``evaluator_groups`` argument has also slightly changed.  To understand how, we
shall have a quick look at how evaluator groups work.

During deployment, the user can supply multiple evaluator groups.  The first
evaluator group is applied to the project directory itself, the second to all
immediate subdirectories, the third to the subdirectories of these, and so
forth.  Since the multi target adapter added one directory level to our project
structure, we thus have to add an additional empty evaluator group before our
AUROC and patient count metrics.

::

    /path/to/deployment/project/dir ← Calculate []
    ├── isMSIH                   ← Calculate [auroc, count]
    │   └── predictions.csv.zip
    └── gender                   ← Calculate [auroc, count]
         └── predictions.csv.zip

After running the deployment script, there will be a ``stats.csv`` in each of
the subdirectories.  While for this few targets it is still feasible to look at
each of these individually, doing so quickly becomes tedious as the number of
targets increases.  We can forgo this hassle by using the ``aggregate_stats``
evaluator, which combines ``stats.csv``s into a single statistics file::

    evaluator_groups=[
        [aggregate_stats],
        [Grouped(auroc), Grouped(count)]
    ]

After rerunning the deployemnt script, the project directory should now have a
``stats.csv`` which combines all the the subdirectories' statistics information
into one handy file.