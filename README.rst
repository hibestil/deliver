=========================
Deliver
=========================
A genetic algorithm implementation for the Multi Depot Vehicle Routing Problem.

Details
------
There are n vehicles roaming around the city and m orders waiting for the delivery.
Vehicles have a limited :

- amount of stock
- and predefined service durations.

Usage
-----
This application accepts input with following formats :

1. JSON (Vehicle, Job, Matrix) Format
    - Json file path can be passed as program argument.
    - Example json file : ``data/input.json``
2. `Cordeau’s Instances <https://github.com/fboliveira/MDVRP-Instances/blob/master/DESCRIPTION.mdL>`_
    - Benchmark parameter needs to be to set. (ie. ``problem = ProblemHelper(path, benchmark=True)``)
Output
-------
Deliver produces a json file shown in ``data\output.json``. And the expected console output is shown below:

.. example-output::
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
            |_ Carried load of this vehicle is :  1
            |_ and goes to these customers respectively :
                |_ customer: 9	demand:1
            |_ Vehicle returns depot 0
            |_ Total duration of this trip is  1209
        ----------------------------------------------------
        Vehicle : 0
            |_ Leaves from depot 0
            |_ Carried load of this vehicle is :  4
            |_ and goes to these customers respectively :
                |_ customer: 4	demand:1
                |_ customer: 7	demand:1
                |_ customer: 8	demand:1
                |_ customer: 5	demand:1
            |_ Vehicle returns depot 0
            |_ Total duration of this trip is  3757
        ----------------------------------------------------
        Vehicle : 2
            |_ Leaves from depot 2
            |_ Carried load of this vehicle is :  2
            |_ and goes to these customers respectively :
                |_ customer: 6	demand:2
            |_ Vehicle returns depot 2
            |_ Total duration of this trip is  462

        Process finished with exit code 0

``
Installation
------------
#. Install the project's development and runtime requirements::

        pip install -r requirements-dev.txt

#. Run the tests::

        paver test_all

Authors
=======

* Halil İbrahim Bestil
