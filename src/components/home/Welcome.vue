<template>
  <div>
    <v-row>
      <v-carousel cycle height="600" hide-delimiter-background show-arrows-on-hover class="mx-auto">
        <v-carousel-item v-for="(item,i) in items" :key="i" :src="item.src">
          <v-row class="fill-height" align="center" justify="start">
            <v-col offset="1">
              <div class="display-3">{{ item.Title }}</div>
              <div class="display-3">{{ item.Title2 }}</div>
            </v-col>
          </v-row>
        </v-carousel-item>
      </v-carousel>
    </v-row>
    <v-row class="pa-6">
      <v-card class="mx-auto" outlined width="max">
        <v-card-text align="center" justify="center">
          <p class="display-3 text--primary">About</p>
          <div class="text--primary">
            <p class="font-weight-light mb-3 mx-auto title text--darken-3 grey--text">
              Real Time Accident Detection using Yolov3 object detection and Image Processing (OpenCV, Keras, TensorFlow and Pytorch).
              Trained using Google Images, 1700 pictures of accidents occurring on roads and 1800 pictures of roads and traffic.
              Custom proximity and collision detection algorithm to ensure accurate detection of accidents.
            </p>
          </div>
        </v-card-text>
      </v-card>
    </v-row>
  </div>
</template>


<script>
export default {
  data() {
    return {
      items: [
        {
          Title: "It works in Real-Time",
          Title2: "using Google Firebase",
          src: "https://cdn.hipwallpaper.com/i/82/72/smnrIB.png",
        },
        {
          Title: "It's Fast",
          src:
            "https://cloud.addictivetips.com/wp-content/uploads/2018/03/internet-speed.jpg",
        },
        {
          Title: "Custom trained Neural-Network",
          src: "https://miro.medium.com/max/1000/1*7wN5t9ILU0fpnhbMX2vtng.jpeg",
        },
      ],
    };
  },
};
</script>