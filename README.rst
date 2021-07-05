.. image:: data/diagram.png
  :alt: Deliver MDVRP Genetic Algorithm System Diagram

=========================
Deliver
=========================
A genetic algorithm implementation for Multi Depot Vehicle Routing Problem.

Details
------
There are n vehicles roaming around the city and m orders waiting for the delivery.
Vehicles have a limited :

- amount of stock
- and predefined service durations.

This project has been developed with an inspiration from the method written by Ombuki-Berman et al.

        Ombuki-Berman, Beatrice, and Franklin T. Hanshar. "Using genetic algorithms for multi-depot vehicle routing." Bio-inspired algorithms for the vehicle routing problem. Springer, Berlin, Heidelberg, 2009. 77-99.
Usage
-----
Create a new enviroment and install required packages:
::
        pip install -r requirements.txt
Run script with this command:
::
        python -m deliver.main -i ./data/input.json --intermediate_prints -o ./data/output.json

To use `Cordeau’s Instances <https://github.com/fboliveira/MDVRP-Instances/blob/master/DESCRIPTION.md>`_ please use related input data and set ``--benchmark_input`` option.
::
        python -m deliver.main -i ./data/p01 --intermediate_prints --benchmark_input -o ./data/output.json

Output
-------
"Deliver" produces a json file shown in ``data\output.json``. And the expected console output is shown below:
::
        [Generation 0] Best score: 0.00013974287311347122 Consistent: True
        [Generation 10] Best score: 0.00018422991893883567 Consistent: True
        .
        .
        .
        [Generation 2470] Best score: 0.00018422991893883567 Consistent: True
        [Generation 2480] Best score: 0.00018422991893883567 Consistent: True
        [Generation 2490] Best score: 0.00018422991893883567 Consistent: True


        Finished training
        Best score: 0.00018422991893883567, best distance: 5428
        -----------------------SUMMARY-----------------------
        Total duration : 5428
        ----------------------------------------------------
        Vehicle : 0
            |_ Leaves from depot 0
            |_ Amount of carried load by this vehicle is :  4
            |_ Goes to these customers respectively :
                |_ customer: 4	demand:1
                |_ customer: 7	demand:1
                |_ customer: 8	demand:1
                |_ customer: 5	demand:1
            |_ Vehicle returns to the depot 0
            |_ Total duration of this trip is  3757
        ----------------------------------------------------
        Vehicle : 1
            |_ Leaves from depot 1
            |_ Amount of carried load by this vehicle is :  1
            |_ Goes to these customers respectively :
                |_ customer: 9	demand:1
            |_ Vehicle returns to the depot 0
            |_ Total duration of this trip is  1209
        ----------------------------------------------------
        Vehicle : 2
            |_ Leaves from depot 2
            |_ Amount of carried load by this vehicle is :  2
            |_ Goes to these customers respectively :
                |_ customer: 6	demand:2
            |_ Vehicle returns to the depot 2
            |_ Total duration of this trip is  462

        Process finished with exit code 0


``

Installation for Development
------------
#. Install the project's development and runtime requirements::

        pip install -r requirements-dev.txt

#. Run the tests::

        paver test_all

Authors
=======

* Halil İbrahim Bestil
