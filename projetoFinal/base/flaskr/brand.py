from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db

bp = Blueprint("brand", __name__)


@bp.route("/")
def index():
    """Show all the brands, most recent first."""
    db = get_db()

    brands = db.execute(
        "SELECT brand, COUNT(*) as numberOfComplains"
        " FROM complain"
        " GROUP BY brand "
        " ORDER BY brand DESC"
    )

    brands = brands.fetchall()

    for row in brands:
        if row == 0:
            print ('no row was found')
            render_template("main/index.html", brands=None)

    return render_template("main/index.html", brands=brands)



@bp.route("/<string:id>/brand")
def brandPage(id):
    """Show all the complains, most recent first."""
    db = get_db()
    complains = db.execute(
        "SELECT *"
        " FROM complain"
        " WHERE brand = ?" 
        " ORDER BY idd DESC",
            (id,),
    ).fetchall()


    return render_template("brand/brandpage.html", brandName=id, complains=complains)



