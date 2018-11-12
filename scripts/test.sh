{ sleep $1; echo one; } &
{ sleep $2; echo two; } &
wait -n
pkill -P $$
echo done

