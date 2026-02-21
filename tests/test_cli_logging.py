import logging

from neb_dynamics.scripts import main_cli


def test_cli_logging_suppresses_geometric_loggers():
    main_cli._configure_cli_logging()

    assert logging.getLogger("geometric").disabled is True
    assert logging.getLogger("geometric.nifty").disabled is True
    assert logging.getLogger("geometric").level == logging.WARNING
    assert logging.getLogger("geometric.nifty").level == logging.WARNING

