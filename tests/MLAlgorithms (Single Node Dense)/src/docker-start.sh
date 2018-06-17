docker exec -it scidb-container sh -c "service postgresql start"
sleep 1m
docker exec -it scidb-container sh -c "printf 'y' | scidb.py initall scidb"
docker exec -it scidb-container sh -c "scidb.py startall scidb"
