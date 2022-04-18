<template>
  <div>
    <iframe :src="url" frameborder="0" style="border:0;" height="900" width="100%"></iframe>
  </div>
</template>


<script>
export default {
  props: ["destination", "origin"],
  data() {
    return {
      url:
        "https://www.google.com/maps/embed/v1/directions?key=AIzaSyB4W3q7EWYYdZYAggTMvDVinjf_zYX8Yr0&origin=" +
        this.origin.latitude +
        "," +
        this.origin.longitude +
        "&destination=" +
        this.destination.latitude +
        "," +
        this.destination.longitude,
    };
  },
};
</script>